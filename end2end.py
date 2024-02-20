import pdb
import logging
logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
import torch
import yaml
from train.training_loop import TrainLoop
from model.classify_model import Net


def main(args, device):
    
    train_dataset, valid_dataset, test_dataset = None, None, None
    batch_size = args.batch_size
    # check! small batch and normal batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    logging.info('len of train loader:{}, len of test loader:{}'.format(0, 0))

    model = Net(args)
    model.to(device)
    TrainLoop(args, model, device, train_loader, valid_loader).run_loop()


if __name__ == '__main__':
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    config = EasyDict(config)

    main(config, device)
