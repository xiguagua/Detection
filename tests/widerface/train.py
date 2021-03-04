import os
import logging
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import (
    default_setup, 
    default_argument_parser, 
    launch,
)
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
    EventStorage,
)
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)

from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    inference_on_dataset,
    print_csv_format,
    DatasetEvaluators,
)

from detectron2.modeling import build_model
import detectron2.utils.comm as comm

from .dataset import register 

logger = logging.getLogger('widerface')

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)



def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
    #     evaluator_list.append(
    #         SemSegEvaluator(
    #             dataset_name,
    #             distributed=True,
    #             output_dir=output_folder,
    #         )
    #     )
    # if evaluator_type in ["coco", "coco_panoptic_seg"]:
    #     evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {}".format(dataset_name)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results



def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def add_centernet_config(cfg):
    """
    Add config for VoVNet.
    """
    _C = cfg

    _C.MODEL.POSERESNETS = CN()
    _C.MODEL.CENTERNET = CN()

    _C.MODEL.POSERESNETS.DEPTH = 18
    _C.MODEL.POSERESNETS.OUT_FEATURES = []

    _C.MODEL.CENTERNET.IN_FEATURES = []
    _C.MODEL.CENTERNET.HM_WEIGHT = 1
    _C.MODEL.CENTERNET.WH_WEIGHT = 0.1
    _C.MODEL.CENTERNET.WH_LOSS = 'l1'
    _C.MODEL.CENTERNET.DOWN_RATIO = 4

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg



def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    import sys
    abspath = os.path.abspath("../")
    # print(abspath)
    sys.path.insert(0, abspath)  
    import xgdetectron.modeling as _

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    _root = os.getenv("WIDERFACE_DATASETS", "datasets")
    register(_root)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
