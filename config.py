from easydict import EasyDict as edict

config = edict()
config.dataset = "webface"
config.img_size = 112
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = False  # False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256  # 64-w 256-g
config.lr = 0.1  # bs is 512
config.output = config.dataset + "-mobilev3-new"

if config.dataset == "glint360k":
    # make training faster
    # our RAM is 256G
    # mount -t tmpfs -o size=140G  tmpfs /train_tmp
    config.rec = '/data/cve_data/glint360/glint360_data/'
    config.num_classes = 360232
    config.num_image = 17091657
    config.num_epoch = 20
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
    config.dropout = 0.0

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [8, 12, 15, 18] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = '/home/xianfeng.chen/datasets/faces_webface_112x112'
    config.num_classes = 10572
    config.num_image = "forget"
    config.num_epoch = 100
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
    config.dropout = 0.4

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 70, 90] if m - 1 <= epoch])
    config.lr_func = lr_step_func

