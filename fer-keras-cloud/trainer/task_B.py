#__author__ = ritvikareddy2
#__date__ = 2018-12-12


import argparse

import trainer.model as model
import trainer.utils as utils

from datetime import datetime


def train(job_dir, train_file, **args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(train_file))
    print('Using logs_path located at {}'.format(logs_path))
    print('-----------------------')

    data = utils.read_data(train_file)

    x_train, y_train = utils.process_data(data, 'Training')
    x_validation, y_validation = utils.process_data(data, 'PublicTest')

    cnn_model = model.get_simple_model('B')

    # num epochs
    epochs = 100

    # batch size
    batch_size = 100

    # run model
    model_history = cnn_model.fit(x_train, y_train, epochs=epochs,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  validation_data=(x_validation, y_validation),
                                  verbose=2)

    print("model trained")

    utils.plot_losses(job_dir, model_history, 'loss_comparison_B.png')

    print("plotted losses")

    cnn_model.save('fer_model_B.h5')

    utils.copy_file_to_gcs(job_dir, 'fer_model_B.h5')

    print("--------------------DONE----------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    train(**arguments)
