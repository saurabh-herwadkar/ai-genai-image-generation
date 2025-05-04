# Imports
import os
import logging



# Set logging level
logging.getLogger().setLevel(logging.INFO)
from ai_genai_image_generation.autoencoder.auto_encoder import run_auto_encoder


# The main program to run
# Your crews related code will come here
def main_run_auto_encoder():

    logging.info("--------Entering main method--------------")

    run_auto_encoder()

    logging.info("---------Leaving main method------------")
    return ""
