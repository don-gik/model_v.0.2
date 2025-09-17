from module import model
from module import trainer
import argparse
import logging

import os



if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", dest="debug", action="store_true")
    parser.add_argument("-t", "--train", dest="train", action="store")

    args = parser.parse_args()


    if not os.path.exists('models'):
        os.mkdir('models')

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

        trainer.encoder_trainer.main()
    elif args.train == "whole":
        logging.info("The whole model is getting prepared for training.")

        trainer.whole_trainer.run()
    elif args.train == "long":
        logging.info("The Long Whole model is getting prepared for training.")

        trainer.long_time_whole.run()
    else:
        logging.info(f"Training Failed. {args.train} is not detected.")