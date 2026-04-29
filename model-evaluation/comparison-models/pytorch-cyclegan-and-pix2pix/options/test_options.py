from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=1_000_000_000, help='how many test images to run')

        # Provide a list of ensemble models
        parser.add_argument('--ensemble_models', type=str, nargs='+', default=[], help='a space-separated list of model names for ensemble testing. Eg: --ensemble_models model1 model2 model3')
        parser.add_argument('--test_name', type=str, help='the name of the current test which results will be saved under.')

        # Optionally output test results as .npy files
        parser.add_argument('--save_npy', action='store_true', default=False, help='if specified, saves the test results as .npy files in addition to images.')

        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
