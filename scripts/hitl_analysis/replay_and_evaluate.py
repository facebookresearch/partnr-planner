# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
import pickle
from multiprocessing import Pool

import habitat.sims.habitat_simulator.sim_utilities as sutils
import magnum as mn
from compute_scores import compute_statistics
from habitat.sims.habitat_simulator.object_state_machine import set_state_of_obj
from hitl_episode import HITLSession, divide_list, init_env

IM_W = 800


def get_metrics_episode_id(env, session, episode_id=0):
    """
    Gets the metrics on each frame of the HITL collected data
    """
    # get/set episode by EID
    hitl_episode = [
        ep for ep in session.episodes if ep.episode_info["episode_id"] == episode_id
    ][0]
    env.current_episode = [
        ep for ep in env._dataset.episodes if ep.episode_id == episode_id
    ][0]

    hitl_ep = hitl_episode.hitl_data["episode"]
    hitl_id = int(hitl_episode.episode_info["episode_id"])
    ep_info = hitl_ep["episode_info"]
    info = {
        "user_id": hitl_episode.hitl_data["users"][0]["connection_record"]["user_id"],
        "episode_info": hitl_episode.episode_info,
        "episode_id": hitl_id,
        "finished": hitl_ep["finished"],
        "wall_clock_time": int(hitl_ep["end_timestamp"])
        - int(hitl_ep["start_timestamp"]),
        "num_steps": len(hitl_episode.hitl_data["frames"]),
        "tutorial": ep_info["is_tutorial"] if "is_tutorial" in ep_info else False,
        "hitl_task_complete": hitl_ep["task_percent_complete"],
        "hitl_task_explantion": hitl_episode.hitl_data["metrics"]["task_explanation"],
    }

    # We should remove this, it is an issue of habitat sim
    for agent_id in range(len(env.sim.agents)):
        env.sim.initialize_agent(agent_id)

    obs = env.reset()

    metrics = []
    ontop_recep = []
    for _id, frame_event in enumerate(hitl_episode.hitl_data["frames"]):
        if len(frame_event) == 0:
            continue

        metrics.append(dict(env.get_metrics().items()))

        obj_ontop = {}
        failed = False
        for object_state in frame_event["object_states"]:
            handle = object_state["object_handle"]
            obj = sutils.get_obj_from_handle(env.sim, handle)
            if obj is None:
                failed = True
                print(
                    f"cannot evaluate episode: object handle {handle} not found in scene."
                    " Is there a mismatch between the HITL episode and dataset episode?"
                )
                break

            obj.translation = mn.Vector3(object_state["position"])
            obj.rotation = mn.Quaternion(
                object_state["rotation"][:-1], object_state["rotation"][-1]
            )

            for user_event in frame_event["users"]:
                event_list = user_event["events"]
                if len(event_list) == 0:
                    continue
                for event in event_list:
                    if event["type"] != "state_change":
                        continue
                    obj = sutils.get_obj_from_handle(env.sim, event["obj_handle"])
                    set_state_of_obj(obj, event["state_name"], event["new_value"])

            recep_ids = sutils.above(env.sim, obj)
            recep_objs = [sutils.get_obj_from_id(env.sim, rid) for rid in recep_ids]

            obj_ontop[object_state["object_handle"]] = [
                ro.handle for ro in recep_objs if ro is not None
            ]
        if failed:
            return info

        try:
            env._task.measurements.update_measures(
                episode=env.current_episode,
                task=env._task,
                observations=obs,
            )
        except Exception as e:
            print(f"cannot evaluate episode: {hitl_id}. Reason: \n{e}\n")

        ontop_recep.append(obj_ontop)

    info["metrics"] = [metrics[-1] if len(metrics) > 0 else []]
    info["ontop"] = ontop_recep
    return info


def eval_session(session):
    """
    Eval a HITL episode
    """
    episode_ids = [episode.episode_info["episode_id"] for episode in session.episodes]
    try:
        cfg_dict = session.episodes[0].hitl_data["session"]["config"]
    except Exception:
        print(len(session.episodes))

    env = init_env(session, episode_ids=episode_ids, cfg_dict=cfg_dict)
    return [get_metrics_episode_id(env, session, eid) for eid in episode_ids]


def process_session_file(info):
    x, y, z = info
    return eval_session(HITLSession(file_list=x, multi=y, hitl_data_file=z))


def batch_process_session_file(
    episodes_path: str, dataset_file: str, is_multi_user: bool, n_cpus: int = 20
) -> None:
    file_list = [
        os.path.join(episodes_path, file) for file in os.listdir(episodes_path)
    ]
    files = divide_list(file_list, n_cpus=n_cpus)
    inputs = [[file, is_multi_user, dataset_file] for file in files]
    all_eps = []
    with Pool(n_cpus) as p:
        all_eps.append(p.map(process_session_file, inputs))

    with open(os.path.join(episodes_path, "best_episodes.pkl"), "wb") as f:
        pickle.dump(all_eps, f)


def show_metrics(episodes_path: str) -> None:
    with open(os.path.join(episodes_path, "best_episodes.pkl"), "rb") as f:
        all_eps = pickle.load(f)

    task_percent_complete = {}
    task_state_success = {}
    skipped = 0
    for proc_eps in all_eps:
        for ep in proc_eps:
            if "metrics" not in ep:
                skipped += 1
                continue

            eid = ep["episode_info"]["episode_id"]
            task_percent_complete[eid] = ep["metrics"][0]["task_percent_complete"]
            task_state_success[eid] = ep["metrics"][0]["task_state_success"]

    print("Num episodes evaluated:", len(task_percent_complete))
    print("Num episodes skipped:", skipped)

    print("\n\n---Success Rate---")
    success = list(task_state_success.values())
    compute_statistics(success)

    print("\n\n---Percent Complete---")
    pc = list(task_percent_complete.values())
    compute_statistics(pc)


if __name__ == "__main__":
    """Replay HITL episodes in sim and evaluate performance."""
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
        default=20,
    )
    parser.add_argument(
        "--multi",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="provide this flag if the collection was multi-user.",
    )
    parser.add_argument(
        "--just-evaluate",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If the episodes have already been processed, you can re-evaluate them with this flag.",
    )
    args = parser.parse_args()
    if not args.just_evaluate:
        batch_process_session_file(
            args.episodes_path, args.dataset_file, bool(args.multi), args.ncpus
        )
    show_metrics(args.episodes_path)
