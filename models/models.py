
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycleCtrl':
        from .cycle_gan_model_Ctrl import CycleGANCtrlModel
        model = CycleGANCtrlModel()
    elif opt.model == 'cycleCtrl_lowlight':
        from .lowlight_model import LowLightModel
        model = LowLightModel()
    elif opt.model == 'cycleCtrl_noise':
        from .noise_model import NoiseModel
        model = NoiseModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
