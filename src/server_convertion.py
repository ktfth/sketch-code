#!/usr/bin/env python
import io
import os
import sys
import tensorflow as tf
from argparse import ArgumentParser
from os.path import basename

from classes.inference.Sampler import *

from PIL import Image

from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response

def build_parser():
  parser = ArgumentParser()
  parser.add_argument('--output_folder', type=str,
                      dest='output_folder', help='dir to save generated gui and html',
                      required=True)
  parser.add_argument('--model_json_file', type=str,
                      dest='model_json_file', help='trained model json file',
                      required=True)
  parser.add_argument('--model_weights_file', type=str,
                      dest='model_weights_file', help='trained model weights file', required=True)
  parser.add_argument('--style', type=str,
                      dest='style', help='style to use for generation', default='default')
  parser.add_argument('--print_generated_output', type=int,
                      dest='print_generated_output', help='see generated GUI output in terminal', default=1)
  parser.add_argument('--print_bleu_score', type=int,
                      dest='print_bleu_score', help='see BLEU score for single example', default=0)
  parser.add_argument('--original_gui_filepath', type=str,
                      dest='original_gui_filepath', help='if getting BLEU score, provide original gui filepath', default=None)

  return parser

def main(argv):
    parser = build_parser()
    options = parser.parse_args(argv[1:])

    def decode(request):
        data = {'success': False}

        if request.method == 'POST':
            if request.POST.__contains__('image'):
                image = request.POST['image'].file.read()
                image = Image.open(io.BytesIO(image))

                output_folder = options.output_folder
                model_json_file = options.model_json_file
                model_weights_file = options.model_weights_file
                style = options.style
                print_generated_output = options.print_generated_output
                print_bleu_score = options.print_bleu_score
                original_gui_filepath = options.original_gui_filepath

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                sampler = Sampler(model_json_path=model_json_file,
                                  model_weights_path = model_weights_file)
                converted = sampler.convert_single_image(output_folder, png_path=image, print_generated_output=print_generated_output, get_sentence_bleu=print_bleu_score, original_gui_filepath=original_gui_filepath, style=style)

                if converted:
                    data['success'] = True

        return data

    with Configurator() as config:
    	config.add_route('decode', '/decode')
    	config.add_view(decode, route_name='decode', renderer='json')
    	app = config.make_wsgi_app()

    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
