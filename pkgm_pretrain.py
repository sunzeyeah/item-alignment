import os
import argparse
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchkge import LinkPredictionEvaluator
from torchkge import TransEModel, PKGMModel
from torchkge import load_ccks
from torchkge import Trainer, MarginLoss
from src.utils import logger


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_name", default="transe_epoch-{}.bin", type=str, help="model saving name",)
    # training
    parser.add_argument("--do_eval", action="store_true", help="是否进行模型验证")
    parser.add_argument("--do_test", action="store_true", help="是否进行模型测试")
    parser.add_argument("--cuda_mode", default="all", help="cuda mode, all or batch")
    parser.add_argument("--train_batch_size", default=2048, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=2048, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting training epoch")
    parser.add_argument("--num_train_epochs", default=1000, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--log_steps", default=None, type=int, help="every n steps, log training process")
    parser.add_argument("--save_epochs", default=1000, type=int, help="every n epochs, save model")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model path")
    # optimization
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--warmup_proportion", default=0.2, type=float, help="warmup proportion in learning rate scheduler")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumualte before performing a backward/update pass.")
    # Graph Embedding
    parser.add_argument("--dim", default=768, type=int, help="dimension of graph embedding")
    parser.add_argument("--margin", default=1.0, type=float, help="maring loss")
    parser.add_argument("--n_neg", default=3, type=int, help="number of negative samples")
    # parser.add_argument("--negative_entities", default=3, type=int, help="number of negative entities")
    # parser.add_argument("--negative_relations", default=3, type=int, help="number of negative relations")
    parser.add_argument("--norm", default="L2", type=str, help="vector norm: L1, L2, torus_L1, torus_L2")
    parser.add_argument("--sampling_type", default="bern", type=str, help="sampling type, Either 'unif' (uniform negative sampling) or "
                                                                               "'bern' (Bernoulli negative sampling)")

    return parser.parse_args()


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main():
    args = get_parser()

    # Load dataset
    kgs = load_ccks(args.data_dir, args.do_eval, args.do_test)
    kg_train = kgs[0]
    logger.info(f"finished loading data")

    # Define the model and criterion
    if "transe" in args.model_name:
        model = TransEModel(args.dim, kg_train.n_ent, kg_train.n_rel,
                            dissimilarity_type=args.norm)
    elif "pkgm" in args.model_name:
        model = PKGMModel(args.dim, kg_train.n_ent, kg_train.n_rel,
                            dissimilarity_type=args.norm)
    else:
        raise ValueError(f"Unsuported model name: {args.model_name}")
    if args.pretrained_model_path is not None:
        state_dict = torch.load(args.pretrained_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    criterion = MarginLoss(args.margin)
    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay, eps=args.adam_epsilon)
    num_train_optimization_steps = int(
        len(kg_train)
        / args.train_batch_size
        / args.gradient_accumulation_steps
    ) * (args.num_train_epochs - args.start_epoch)
    num_warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_optimization_steps)

    # Start Training
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    use_cuda = args.cuda_mode if torch.cuda.is_available() else None
    model_save_path = os.path.join(args.output_dir, args.model_name)
    trainer = Trainer(model, criterion, kg_train, args.num_train_epochs,
                      args.train_batch_size, optimizer=optimizer, scheduler=scheduler,
                      model_save_path=model_save_path, sampling_type=args.sampling_type,
                      n_neg=args.n_neg, use_cuda=use_cuda, fp16=args.fp16, scaler=scaler,
                      log_steps=args.log_steps, start_epoch=args.start_epoch,
                      save_epochs=args.save_epochs, gradient_accumulation_steps=args.gradient_accumulation_steps)
    trainer.run()

    # Evaluation
    if args.do_test:
        if args.do_eval:
            kg_test = kgs[2]
        else:
            kg_test = kgs[1]
        evaluator = LinkPredictionEvaluator(model, kg_test)
        evaluator.evaluate(args.eval_batch_size)
        evaluator.print_results()


if __name__ == "__main__":
    main()
