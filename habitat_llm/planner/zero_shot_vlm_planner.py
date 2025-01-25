from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import PIL
from PIL import ImageDraw, ImageFont

from habitat_llm.planner import ZeroShotReactPlanner
from habitat_llm.utils import geometric

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model.world_graph import WorldGraph

from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer

from habitat_llm.llm.instruct.utils import pil_image_to_data_url


# TODO: move into drawing utils
def add_text(image: "PIL.Image", coords: List[tuple[float, float]], texts: List[str]):
    assert len(coords) == len(texts)
    draw = ImageDraw.Draw(image)
    # Load a default font
    font = ImageFont.load_default()
    for coord, text in zip(coords, texts):
        _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)

        # Define padding for the textbox around the text
        padding = 5
        x, y = coord
        box_left = x - padding
        box_top = y - padding
        box_right = x + text_width + padding
        box_bottom = y + text_height + padding

        # Draw the rectangle (textbox)
        draw.rectangle(
            [box_left, box_top, box_right, box_bottom], fill="white", outline="black"
        )

        draw.text(coord, text, (0, 0, 0))


def get_name_and_coords(entities):
    names = []
    coords = []
    for entity in entities:
        names.append(entity.name)
        coords.append(np.array(entity.properties["translation"])[None, ...])
    if len(coords) > 0:
        coords = np.concatenate(coords, 0)
    return names, coords


class ZeroShotVLMPlanner(ZeroShotReactPlanner):
    """
    This class builds the prompt for the, zero shot llm react planner format.
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ) -> None:
        """
        Initialize the ZeroShotReactPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        self.dbv = DebugVisualizer(env_interface.sim, resolution=(480, 640))
        super().__init__(plan_config, env_interface)

    def prepare_prompt(
        self,
        input_instruction,
        world_graph,
        observations,
        legend_to_name_mapping=None,
        **kwargs,
    ):
        """Prepare the prompt for the LLM, both by adding the input
        and the agent descriptions"""
        _, params = super().prepare_prompt(
            input_instruction, world_graph, should_format=False
        )
        if legend_to_name_mapping is None:
            legend_to_name_mapping = {}

        if "{objects_seen}" in self.prompt:
            # only designed for the decentralized setting
            assert len(self.agents) == 1
            world_description = ""
            for legend_name in sorted(legend_to_name_mapping.keys()):
                world_description += (
                    f"{legend_name}: {legend_to_name_mapping[legend_name]}\n"
                )
            world_description = world_description.strip()
            params["objects_seen"] = world_description

        return self.prompt.format(**params), params

    def replan(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ):
        """
        Replan a high level action using the LLM/VLM
        """

        # Get coordinates of objects and furniture
        curr_graph = world_graph[self.agents[0].uid]
        furnitures = curr_graph.get_all_furnitures()
        objects = curr_graph.get_all_objects()
        sim = self.env_interface.env.env.env._env.sim

        # We will use a top down view and project all the objects in the scene
        top_obs = self.dbv.peek("scene")
        obs_image = top_obs.get_image()
        obs_camera = self.dbv.sensor._sensor_object.render_camera
        furniture_name, furniture_coords = get_name_and_coords(furnitures + objects)

        # Get a legend from object ids to object names
        imsize = obs_image.size
        legend_to_name_mapping: Dict[int, str] = {}
        for entity_name in furniture_name:
            ind = len(legend_to_name_mapping) + 1
            legend_to_name_mapping[ind] = entity_name

        # Project the coordinates according to the camera parameters
        if len(furniture_coords) > 0:
            im_coord = geometric.project_to_im_coordinates(
                furniture_coords,
                obs_camera.camera_matrix,
                obs_camera.projection_matrix,
                np.array(imsize),
            )
            num_furniture = im_coord.shape[0]
            # Draw numbers according to the coordinates
            add_text(
                obs_image,
                list(im_coord),
                [str(u + 1) for u in range(num_furniture)],
            )

        # Draw the robot in the frame
        spot = sim.agents_mgr[0].articulated_agent
        tx_spot = np.array(spot.ee_transform().translation)[None, ...]
        im_coord = geometric.project_to_im_coordinates(
            tx_spot,
            obs_camera.camera_matrix,
            obs_camera.projection_matrix,
            np.array(imsize),
        )
        add_text(obs_image, list(im_coord), ["Me"])

        current_prompt, self.params = self.prepare_prompt(
            instruction,
            world_graph[self._agents[0].uid],
            observations=observations,
            legend_to_name_mapping=legend_to_name_mapping,
        )
        if self.curr_prompt == "":
            self.curr_prompt = current_prompt
        prompt = [
            ("text", current_prompt),
            ("image", pil_image_to_data_url(obs_image)),
        ]
        llm_response = self.llm.generate(prompt)

        # Format the response
        # This removes extra text followed by end expression when needed.
        llm_response = self.format_response(llm_response, self.end_expression)

        info = {"llm_response": llm_response}
        return info

    def reset(self) -> None:
        """
        Reset the planner state.
        """
