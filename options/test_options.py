from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--N_search', type=int, default=7)
        self.parser.add_argument('--resume_ckpt', type=int, default=1)
        self.parser.add_argument('--lowerb', type=float, default=0., help='lower bound of enhance intensity')
        self.parser.add_argument('--higherb', type=float, default=1., help='higher bound of enhance intensity')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        self.parser.add_argument('--save_path', type=str, default='./results', help='where to save the predicted images')
        self.parser.add_argument('--config', type=str, default='', help='Path to the config file.')
        self.parser.add_argument('--inf_type', type=str, default='exhaustive', help='ternary_search or exhaustive')
        self.parser.add_argument('--input_path', type=str, default='', help='input_path')
        self.parser.add_argument('--gt_path', type=str, default='', help='gt_path. for ternary search')
        self.parser.add_argument('--ts_critic', type=str, default='psnr', help='metric that guides ternary search')
        self.parser.add_argument('--tag', type=str, default='', help='recommend as S2W/W2S/C2D/D2C')
        self.isTrain = False
