# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to run multinode training with submitit.
"""
import argparse
import os
from pathlib import Path

import submitit


def parse_args():
    parser = argparse.ArgumentParser("Submitit for pycls")
    parser = argparse.ArgumentParser(description="Config file options.")
    parser.add_argument("--cfg", dest="cfg_file", help="Config file location", required=True, type=str)
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2000, type=int, help="Duration of the job")
    parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    parser.add_argument("opts", help="See pycls/core/config.py for all options", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments/pycls")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import pycls.core.config as config
        import pycls.core.distributed as dist
        import pycls.core.trainer as trainer
        from pycls.core.config import cfg

        job_env = submitit.JobEnvironment()
        output_dir = str(self.args.job_dir).replace("%j", str(job_env.job_id))
        self.args.opts = ["OUT_DIR", output_dir, "WANDB.RUN_ID", str(job_env.job_id)] + self.args.opts

        config.load_cfg_fom_args("Train a classification model.", args=self.args)
        config.assert_and_infer_cfg()
        cfg.freeze()
        
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)

    def checkpoint(self):
        import submitit

        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=1,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs,
    )

    executor.update_parameters(name="pycls")

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
