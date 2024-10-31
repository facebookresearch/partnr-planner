# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
from multiprocessing import Pool

import matplotlib.animation as animation
import numpy as np
import pandas as pd
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from hitl_episode import HITLEpisode, HITLSession, divide_list, init_env
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from habitat_llm.utils.sim import find_receptacles

IM_W = 800


def print_clutter_separation(episode):
    n_objects = len(episode["name_to_receptacle"])
    n_clutter = 0
    for state_cfg in episode["info"]["initial_state"]:
        if "name" in state_cfg:  # is clutter
            n_clutter += int(state_cfg["number"])

    obj_hashes = list(episode["name_to_receptacle"])
    return obj_hashes[: (n_objects - n_clutter)]


def load_metadata(metadata_dict):
    """
    This method loads the metadata about objects and receptacles.
    This data will typically include tags associated with these objects such as type, color etc.
    """

    # Make sure that metadata_dict is not None
    if not metadata_dict:
        raise ValueError("Cannot load metadata from None")

    # Fetch relevant paths
    metadata_folder = metadata_dict["metadata_folder"]
    object_metadata_path = os.path.join(metadata_folder, metadata_dict["obj_metadata"])
    static_object_metadata_path = os.path.join(
        metadata_folder, metadata_dict["staticobj_metadata"]
    )

    # Make sure that the paths are valid
    if not os.path.exists(object_metadata_path):
        raise Exception(f"Object metadata file not found, {object_metadata_path}")
    if not os.path.exists(static_object_metadata_path):
        raise Exception(
            f"Receptacle metadata file not found, {static_object_metadata_path}"
        )

    # Read the metadata files
    df_static_objects = pd.read_csv(static_object_metadata_path)
    df_objects = pd.read_csv(object_metadata_path)

    # Rename some columns
    df1 = df_static_objects.rename(columns={"id": "handle", "main_category": "type"})
    df2 = df_objects.rename(columns={"id": "handle", "clean_category": "type"})

    # Drop the rest of the columns in both DataFrames
    df1 = df1[["handle", "type"]]
    df2 = df2[["handle", "type"]]

    # Merge the two data frames
    union_df = pd.concat([df1, df2], ignore_index=True)

    return union_df


def get_object_property_from_metadata(handle, prop, metadata):
    """
    This method returns value of the requested property using metadata file.
    For example, this could be used to extract the semantic type of any object
    in HSSD. Not that the property should exist in the metadata file.
    """
    # Declare default
    property_value = "unknown"

    # get hash from handle
    handle_hash = handle.rpartition("_")[0]

    # Use loc to locate the row with the specific key
    object_row = metadata.loc[metadata["handle"] == handle_hash]

    # Extract the value from the object_row
    if not object_row.empty:
        # Make sure the property value is not nan or empty
        if object_row[prop].notna().any() and (object_row[prop] != "").any():
            property_value = object_row[prop].values[0]
    else:
        raise ValueError(f"Handle {handle} not found in the metadata.")

    return property_value


def get_furniture_property_from_metadata(handle, prop, metadata):
    """
    This method returns value of the requested property using metadata file.
    For example, this could be used to extract the semantic type of any object
    in HSSD. Not that the property should exist in the metadata file.
    """
    # Declare default
    property_value = "unknown"

    # get hash from handle
    # handle_hash = handle.split(".", 1)[0] if "." in handle else handle.split("_", 1)[0]
    handle_hash = handle.split(".")[0] if "." in handle else handle.rpartition("_")[0]

    # Use loc to locate the row with the specific key
    object_row = metadata.loc[metadata["handle"] == handle_hash]

    # Extract the value from the object_row
    if not object_row.empty:
        # Make sure the property value is not nan or empty
        if object_row[prop].notna().any() and (object_row[prop] != "").any():
            property_value = object_row[prop].values[0]
    else:
        raise ValueError(f"Handle {handle} not found in the metadata.")

    return property_value


def get_fur_dict(sim, verbose: bool = False, metadata=None):
    """
    Adds all furniture and corresponding receptacles to the graph during graph initialization
    """
    # Make sure that sim is not None
    if not sim:
        raise ValueError("Trying to load furniture from sim, but sim was None")

    # Make sure that the metadata is not None
    if metadata is None:
        raise ValueError("Trying to load furniture from sim, but metadata was None")

    # Load rigid and articulated object managers
    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()

    # Get the list of receptacles from sim
    # This list is maintained as a state of this class because
    # its computationally expensive to generate (~0.3 sec) and
    # need to be used elsewhere in the code
    receptacles = find_receptacles(sim)

    dict_fur = {}
    # Iterate through receptacles and populate the furniture and
    # receptacle nodes in the graph.

    for _i, receptacle in enumerate(receptacles):
        # Get the receptacle handle and articulation type from the surface
        furniture_sim_handle = receptacle.parent_object_handle
        is_articulated = receptacle.parent_link is not None

        # Conditionally add furniture to the gt_graph
        furniture_type = get_furniture_property_from_metadata(
            furniture_sim_handle, "type", metadata
        )

        # Generate name for furniture
        furniture_name = f"{furniture_type}"

        # Get furniture translation
        om = aom if is_articulated else rom

        translation = list(om.get_object_by_handle(furniture_sim_handle).translation)
        om.get_object_by_handle(furniture_sim_handle)
        # breakpoint()
        # Create properties dict
        dict_fur[furniture_sim_handle] = (furniture_name, translation)

        # Get furnitre name
        # furniture_name = self.sim_handle_to_name[furniture_sim_handle]

        # Add rec name to handle mapping
        # self.sim_handle_to_name[receptacle.unique_name] = rec_name

    return dict_fur


def cam_to_im(sensor, point):
    im_width = IM_W
    if sensor is not None:
        sensor = sensor._sensor_object
        cam_matrix = sensor.render_camera.camera_matrix
        proj_matrix = sensor.render_camera.projection_matrix
        hfov = sensor.hfov
    else:
        raise Exception

    fov = float(hfov) * np.pi / 180
    fs = 1 / (2 * np.tan(fov / 2.0))

    point_homog = np.concatenate([point, np.ones((point.shape[0], 1))], 1)

    point_cam_coord = (np.array(cam_matrix) @ point_homog.transpose()).transpose()
    point_cam_coord = (np.array(proj_matrix) @ point_cam_coord.transpose()).transpose()

    im_coord = fs * point_cam_coord / point_cam_coord[:, [-2]]
    im_coord += 0.5
    im_coord[:, 1] = 1 - im_coord[:, 1]
    im_coord = im_coord[:, :2] * im_width
    return im_coord


class EpisodePlotter:
    def __init__(
        self,
        delta_t=0.5,
        multi_agent=False,
        result=None,
        filepath="",
        metadata=None,
        sensor=None,
        img=None,
        handles_rec=None,
        furniture_dict=None,
    ):
        self.delta_t = delta_t
        self.size_agent = 200
        self.size_objects = 75
        c1 = np.array(plt.cm.rainbow(0))[None, ...]
        c2 = np.array(plt.cm.rainbow(0.3))[None, ...]
        self.multi_agent = multi_agent
        self.res = result
        self.filepath = filepath
        self.metadata = metadata
        self.sensor = sensor
        self.img = img
        self.handles_rec = [] if handles_rec is None else handles_rec
        self.furniture_dict = {} if furniture_dict is None else furniture_dict

        self.num_agents = 2
        if not multi_agent:
            self.num_agents = 1
        self.color_agents_2 = [
            np.array([0, 0, 0, 1.0])[None, ...],
        ] * self.num_agents
        self.color_agents = [c1, c2] if multi_agent else [c1]

    def start_plot(self):
        self.fig, ax = plt.subplots(figsize=(10, 10))
        # ax.set_xlim(-20, 20)
        # ax.set_ylim(-20, 20)
        if self.img is not None:
            ax.imshow(self.img)

        self.scat_plot = ax.scatter([], [], label="objects")
        self.ax = ax
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # Put furniture
        try:
            fur_coords = [
                np.array(self.furniture_dict[handle_fur][1])[None, ...]
                for handle_fur in self.handles_rec
            ]
        except Exception:
            fur_coords = []
        if len(fur_coords) > 0:
            self.handles_name = [
                self.furniture_dict[handle_fur][0] for handle_fur in self.handles_rec
            ]
            fur_coords = np.concatenate(fur_coords, 0)

            xy_coord = cam_to_im(self.sensor, fur_coords)
            x_fur = xy_coord[:, 0]
            y_fur = xy_coord[:, 1]
            scat_plot_fur = ax.scatter([x_fur], [y_fur], marker="x")

            sizes = [self.size_objects * 1.5] * len(self.handles_rec)
            num_obj = len(self.handles_rec)
            colors = [
                np.array(plt.cm.rainbow((i + 1) / num_obj))
                for i in range(len(self.handles_rec))
            ]
            for col in colors:
                col[-1] = 1.0
            scat_plot_fur.set_sizes(sizes)
            scat_plot_fur.set_color(colors)

            width2 = 600
            height2 = 250
            gap = 20
            fs = 12

            width_legend = np.array([width2 - 10] * len(self.handles_rec))
            height_legend = np.array(
                [(height2 + i * gap) for i in range(len(self.handles_rec))]
            )
            self.legend_text = [
                plt.text(
                    width2,
                    height_legend[i] + 5,
                    handle_goal,
                    {"fontsize": fs, "color": "white"},
                )
                for i, handle_goal in enumerate(self.handles_name)
            ]

            leg = ax.scatter([width_legend], [height_legend], marker="x")
            leg.set_color(colors)

        # Legend

    def gen_animation_plot(self, hitl_episode):
        def init():
            self.scat_plot.set_offsets(np.empty((0, 2)))

            curr_episode_step = self.frames_data[0]
            # obj_names = [
            #     obj["object_handle"] for obj in curr_episode_step["object_states"]
            # ]
            # first add task objects, then add clutter objects
            obj_names = []
            obj_states = curr_episode_step["object_states"]
            for obj in obj_states:
                obj_handle = obj["object_handle"]
                if obj_handle in self.handles_goal:
                    obj_names.append(obj_handle)

            for obj in obj_states:
                obj_handle = obj["object_handle"]
                if obj_handle not in self.handles_goal:
                    obj_names.append(obj_handle)

            num_obj = len(obj_names)
            self.color_objects = [None for _ in range(num_obj)]
            self.color_objects_goal = []
            nobj2 = 0

            for i in range(num_obj):
                if obj_names[i] not in self.handles_goal:
                    self.color_objects[i] = np.array([0.6, 0.6, 0, 0.7])[None, ...]
                else:
                    self.color_objects[i] = np.array(plt.cm.rainbow((i + 1) / num_obj))[
                        None, ...
                    ]
                    # self.color_objects[i] = np.array([1.0, 1.0, 1.0, 1.0])[None, ...]
                    self.color_objects_goal.append(self.color_objects[i])
                    nobj2 += 1

            self.sizes = np.array(
                [self.size_agent] * self.num_agents
                + [self.size_objects] * num_obj
                + [50] * self.num_agents
                + [self.size_objects] * nobj2
            )

            self.colors = np.concatenate(
                self.color_agents
                + self.color_objects
                + self.color_agents_2
                + self.color_objects_goal,  # legend for goal objects
                0,
            )
            task_str = self.episode_info["instruction"]

            def insert_newlines(text, interval):
                return "\n".join(
                    text[i : i + interval] for i in range(0, len(text), interval)
                )

            task_str = insert_newlines(task_str, 100)
            self.texts_interest["task"].set_text(f"Task: {task_str}")

            # result_str = "%Complete: {:.2f}\nState Success {:.2f}  \nNo Prop %Complete: {:.2f}\nNo Prop State Success {:.2f}".format(
            #     self.res[0][0], self.res[0][1], self.res[1][0], self.res[1][1]
            # )
            result_str = "%Complete: {:.2f}\n Success {:.2f}".format(
                self.res[0][0] > 0.99,
                self.res[0][0],
            )

            self.texts_interest["results"].set_text(result_str)
            explain_str = self.res[1][0]
            if self.res[0][0] < 0.99:
                self.texts_interest["explaination"].set_text(explain_str)
            return (self.scat_plot,)

        def make_frame(ind):
            """
            Plot a timestamp of the recorded episode
            Return the plot and the next frame where to look for that timestamp
            """
            curr_frame = self.frame_indices[ind]
            curr_time = self.times[ind]
            curr_episode_step = self.frames_data[curr_frame]

            # Get coordinates of agents and objects
            arrays = [[0, 0, 0.2], [0.2, 0, 0]]
            rotation_vec = [
                Rotation.from_quat(agent_state["rotation"]).apply(arrays[i])[None, ...]
                for i, agent_state in enumerate(curr_episode_step["agent_states"])
            ]
            rec_coords = [
                np.array(agent_state["position"])[None, ...]
                for agent_state in curr_episode_step["agent_states"]
            ]

            obj_states = curr_episode_step["object_states"]
            obj_coords = []
            for obj in obj_states:
                if obj["object_handle"] in self.handles_goal:
                    obj_coords.append(np.array(obj["position"])[None, ...])

            for obj in obj_states:
                if obj["object_handle"] not in self.handles_goal:
                    obj_coords.append(np.array(obj["position"])[None, ...])

            # obj_coords = [
            #     np.array(obj["position"])[None, ...]
            #     for obj in curr_episode_step["object_states"]
            # ]
            obj_names = [
                obj["object_handle"] for obj in curr_episode_step["object_states"]
            ]
            len(obj_names)
            # self.color_objects = [None for _ in range(num_obj)]
            # self.color_objects_goal = []

            # Get cooridnates of grabbed objects
            grasped_objs = [
                str(agent_state["held_object"])
                for agent_state in curr_episode_step["users"]
            ]

            events = [
                self.parse_events(user_event) for user_event in self.events_log[ind]
            ]
            obj_coords = rec_coords + obj_coords + rotation_vec
            obj_coords = np.concatenate(obj_coords, 0)
            num_agents = len(rotation_vec)
            # breakpoint()
            obj_coords[-num_agents:, ...] += obj_coords[:num_agents, ...]
            # We drop one coordinate
            # TODO: later we can use camera paramerers intead to show an image and project.
            # xy_coord = obj_coords[:, [0, 2]]
            xy_coord = cam_to_im(self.sensor, obj_coords)
            # breakpoint()

            x = xy_coord[:, 0]
            y = xy_coord[:, 1]
            # breakpoint()
            # ipdb.set_trace()
            x = np.concatenate([x, self.width_legend], 0)
            y = np.concatenate([y, self.height_legend], 0)
            # breakpoint()
            # Text size

            self.scat_plot.set_offsets(np.c_[x, y])
            self.scat_plot.set_sizes(self.sizes)
            self.scat_plot.set_color(self.colors)
            # breakpoint()
            self.texts_interest["time"].set_text(f"Time: {curr_time:.1f}")
            self.texts_interest["event"].set_text(f"Event: {curr_frame:.1f}")
            # breakpoint()
            self.texts_interest["grabbed"][0].set_text(f"human_grab: {grasped_objs[0]}")
            self.texts_interest["actions"][0].set_text(f"human: {events[0]}")
            if self.multi_agent and len(grasped_objs) > 1:
                self.texts_interest["grabbed"][1].set_text(
                    f"robot_grab: {grasped_objs[1]}"
                )
                self.texts_interest["actions"][1].set_text(f"robot: {events[1]}")
            return (self.scat_plot,)

        ani = animation.FuncAnimation(
            self.fig,
            make_frame,
            init_func=init,
            frames=len(self.frame_indices),
            blit=True,
        )
        ep_name = str(hitl_episode.hitl_data["episode"]["episode_id"])
        print(f"Saving... {ep_name}")
        save_dir = os.path.join(self.filepath, "videos")
        os.makedirs(save_dir, exist_ok=True)

        increment = 1
        save_file = os.path.join(save_dir, ep_name)
        while os.path.exists(f"{save_file}.mp4"):
            save_file = f"{save_file}_{increment}"
            increment += 1
        save_file += ".mp4"
        print(f"Saving video file {save_file}")
        ani.save(save_file)

    def plot_episode(self, hitl_episode: HITLEpisode):
        self.start_plot()
        (
            self.frame_indices,
            self.times,
            self.events_log,
        ) = hitl_episode.sample_frames_at_frequency(delta_t=self.delta_t)
        self.frames_data = [
            frame for frame in hitl_episode.hitl_data["frames"] if len(frame) != 0
        ]
        self.episode_info = hitl_episode.episode_info
        self.names_interest = ["human"]
        if self.multi_agent:
            self.names_interest += ["robot"]
        self.names_interest += [
            str(obj["object_id"]) for obj in self.frames_data[0]["object_states"]
        ]
        width = 10
        width2 = 600
        height = 20
        height2 = 80
        gap = 20
        fs = 12

        handles_goal = []
        for proposition in self.episode_info["evaluation_propositions"]:
            if "object_handles" in proposition["args"]:
                handles_goal += proposition["args"]["object_handles"]

        # import ipdb; ipdb.set_trace()
        handles_goal = print_clutter_separation(self.episode_info)
        handles_goal = list(set(handles_goal))
        self.handles_goal = handles_goal
        self.texts_interest = {
            "names_interest": [
                plt.text(0, 0, "", {"fontsize": 10, "color": "yellow"})
                for _ in range(len(self.names_interest))
            ],
            "task": plt.text(
                width, height + 1 * gap, "Task: ", {"fontsize": fs, "color": "yellow"}
            ),
            "time": plt.text(
                width, height + 3 * gap, "Time: ", {"fontsize": fs, "color": "yellow"}
            ),
            "event": plt.text(
                width, height + 4 * gap, "Time: ", {"fontsize": fs, "color": "yellow"}
            ),
            "grabbed": [
                plt.text(
                    width,
                    height + 5 * gap,
                    "Event: ",
                    {"fontsize": fs, "color": "yellow"},
                ),
                plt.text(
                    width,
                    height + 6 * gap,
                    "Event: ",
                    {"fontsize": fs, "color": "yellow"},
                ),
            ],
            "actions": [
                plt.text(width, height + 7 * gap, "Event: ", {"fontsize": fs}),
                plt.text(width, height + 8 * gap, "Event: ", {"fontsize": fs}),
            ],
            "results": plt.text(
                width, height + 9 * gap, "Event: ", {"fontsize": fs, "color": "yellow"}
            ),
            "explaination": plt.text(
                width,
                height + 35 * gap,
                "Explanation: ",
                {"fontsize": fs, "color": "yellow"},
            ),
        }
        self.width_legend = np.array([width2 - 10] * len(self.handles_goal))
        self.height_legend = np.array(
            [(height2 + i * gap) for i in range(len(self.handles_goal))]
        )

        real_obj_name = []
        for obj_name in handles_goal:
            obj_type = get_object_property_from_metadata(
                obj_name, "type", self.metadata
            )
            real_obj_name.append(obj_type)

        self.legend_text = [
            plt.text(
                width2,
                self.height_legend[i] + 5,
                handle_goal,
                {"fontsize": fs, "color": "white"},
            )
            for i, handle_goal in enumerate(real_obj_name)
        ]
        self.gen_animation_plot(hitl_episode)

    def parse_event(self, event_dict):
        """
        Get a string for an event, indicating whether the agent picked place or opened close an object
        """
        # print("event dict ", event_dict)
        if event_dict["type"] in ["pick", "open", "close"]:
            event_str = f"{event_dict['type']} {event_dict['obj_id']}"
        elif event_dict["type"] in ["place"]:
            event_str = f"{event_dict['type']} {event_dict['obj_id']},{event_dict['receptacle_id']}"
        else:
            event_str = ""
        return event_str

    def parse_events(self, event_list):
        return ", ".join([self.parse_event(event) for event in event_list])


def visualize_episode(
    hitl_episode,
    delta_t=0.5,
    multi=False,
    sensor=None,
    img=None,
    env=None,
    metadata=None,
    dataset_path="",
):
    furniture_dict = get_fur_dict(env.sim, metadata=metadata)

    task_explanation = ""
    if (
        "metrics" in hitl_episode.hitl_data
        and "task_explanation" in hitl_episode.hitl_data["metrics"]
    ):
        task_explanation = hitl_episode.hitl_data["metrics"]["task_explanation"]

    res = [
        [hitl_episode.hitl_data["episode"]["task_percent_complete"]],
        [task_explanation],
    ]

    # Get receptacle handles
    handles_rec = []
    for proposition in hitl_episode.episode_info["evaluation_propositions"]:
        if "receptacle_handles" in proposition["args"]:
            handles_rec += proposition["args"]["receptacle_handles"]

    plotter = EpisodePlotter(
        delta_t=delta_t,
        multi_agent=multi,
        result=res,
        filepath=dataset_path,
        metadata=metadata,
        sensor=sensor,
        img=img,
        handles_rec=set(handles_rec),
        furniture_dict=furniture_dict,
    )
    plotter.plot_episode(hitl_episode)


def process_session_file(info):
    file_list, multi, multi_plot, dataset_path, hitl_data_file = info
    print(f"visualizing {file_list}")

    session = HITLSession(file_list, multi=multi, hitl_data_file=hitl_data_file)
    if "config" in session.episodes[0].hitl_data["session"]:
        cfg_dict = session.episodes[0].hitl_data["session"]["config"]

    episode_ids = [episode.episode_info["episode_id"] for episode in session.episodes]
    env = init_env(session, episode_ids=episode_ids, cfg_dict=cfg_dict)

    metadata_dict = {
        "metadata_folder": "data/fpss/metadata/",
        "obj_metadata": "object_categories_filtered.csv",
        "room_objects_json": "room_objects.json",
        "staticobj_metadata": "fpmodels-with-decomposed.csv",
    }
    metadata = load_metadata(metadata_dict)

    for episode_id in episode_ids:
        # get/set episode by EID
        hitl_episode = [
            ep for ep in session.episodes if ep.episode_info["episode_id"] == episode_id
        ][0]
        env.current_episode = [
            ep for ep in env._dataset.episodes if ep.episode_id == episode_id
        ][0]

        # We should remove this, it is an issue of habitat sim
        for agent_id in range(len(env.sim.agents)):
            env.sim.initialize_agent(agent_id)

        dbv = DebugVisualizer(env.sim, resolution=(IM_W, IM_W))
        env.reset()
        dbo = dbv.peek("scene")
        img = np.array(dbo.get_image())[:, :, :3]
        sensor = dbv.sensor

        print("start plot...")
        try:
            visualize_episode(
                hitl_episode,
                multi=multi_plot,
                sensor=sensor,
                img=img,
                env=env,
                metadata=metadata,
                dataset_path=dataset_path,
            )
        except Exception as e:
            print(f"failed to visualize episode {episode_id}. Reason:\n{e}")


def batch_process_session_file(
    episodes_path: str, dataset_file: str, multi: bool, multi_plot: bool, n_cpus: int
) -> None:
    file_list = [
        os.path.join(episodes_path, file) for file in os.listdir(episodes_path)
    ]
    files = divide_list(file_list, n_cpus=n_cpus)
    inputs = [[f, multi, multi_plot, episodes_path, dataset_file] for f in files]
    # process_session_file(inputs[0])
    with Pool(n_cpus) as p:
        p.map(process_session_file, inputs)


if __name__ == "__main__":
    """Generate videos of HITL episode rollouts"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-path",
        type=str,
        required=False,
        default="data/hitl_data/2024-10-02-object-states/p5_single_train_10k/processed/best/",
        help="path to a flat directory containing [eid].json.gz collection files",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        required=False,
        default="data/hitl_data/2024-10-02-object-states/p5_single_train_10k/2024_09_16_train_hitl_10k.json.gz",
        help="path to the episode dataset used in this collection",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--multi",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="provide this flag if the collection was multi-user.",
    )
    parser.add_argument(
        "--multi-plot",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="",
    )
    args = parser.parse_args()

    batch_process_session_file(
        args.episodes_path,
        args.dataset_file,
        bool(args.multi),
        bool(args.multi_plot),
        args.ncpus,
    )
