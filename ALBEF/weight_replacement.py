import torch
import argparse
import ruamel_yaml as yaml
import models.low_rank as low_rank

from transformers import logging
from params import parser_add_data_arguments
from models.model_pretrain import ALBEF


def main(args, config):
    model = ALBEF(config=config,
                  text_encoder=args.text_encoder,
                  init_deit=False)
    # model = model.cuda()

    checkpoint = torch.load(args.original_ckpt, map_location='cpu')
    print("load checkpoint from {}".format(args.original_ckpt))
    model.load_state_dict(checkpoint['model'])

    num_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params before low rank: {:.3f} M".format(num_params_before / 10e6))

    module_lr = low_rank.ModuleLowRank(name_omit=args.name_omit)
    model = module_lr(model)

    num_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params after low rank: {:.3f} M\n".format(num_params_after  / 10e6))
    print("Compression rate is {:.3f}".format(num_params_before / num_params_after))

    checkpoint['model'] = model.state_dict()
    torch.save(checkpoint, args.low_rank_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser_add_data_arguments(parser)
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    args = parser.parse_args()

    logging.set_verbosity_error() # avoid annoying log
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(args, config)
