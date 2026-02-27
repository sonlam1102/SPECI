from generate_caption import do_generate_full_caption
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sample', default=3, type=int)
    parser.add_argument('--version', default=1, type=int)
    parser.add_argument('--task', default="none", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    if args.task == "make_instruction":
        do_generate_full_caption(args.version, args.num_sample)
    elif args.task == "make_caption":
        pass
    else:
        print("No task specified.")
