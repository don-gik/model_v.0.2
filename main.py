from module import model
from module import trainer
import argparse
import logging



if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", dest="debug", action="store_true")
    parser.add_argument("-t", "--train", dest="train", action="store")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            format='[%(asctime)s][%(levelname)s] : %(message)s',
            level=logging.DEBUG,
            datefmt='%m/%d/%Y %I:%M:%S %p',
        )

        logging.debug("Debug mode activated.")
    else:
        logging.basicConfig(
            format='[%(asctime)s][%(levelname)s] : %(message)s',
            level=logging.INFO,
            datefmt='%m/%d/%Y %I:%M:%S %p',
        )

    if args.train == "encoder":
        logging.info("Encoder is getting prepared for training.")
    elif args.train == "decoder":
        logging.info("Decoder is getting prepared for training.")
    elif args.train == "whole":
        logging.info("The whole model is getting prepared for training.")
    else:
        logging.info(f"Training Failed. {args.train} is not detected.")