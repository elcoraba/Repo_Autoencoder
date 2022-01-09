## in WORK - depending on train()
from network.autoencoder import Autoencoder
from train import train

ae_p = Autoencoder()
ae_v = Autoencoder()

P_PATH = DIR + 'autoencoder_p.pth'
V_PATH = DIR + 'autoencoder_v.pth'

files_ = glob.glob( DIR + "*" + "b" )

aep_losses = train(ae_p, P_PATH, to_train="pos", files_=files_,
                   num_epochs=14, min_batch_size=256, 
                   trainlogdir = DIR + "trainlog_pos.txt", vallogdir = DIR + "vallog_pos.txt",
                   train_lossacc_dir=DIR + "aep_train_lossacc.pkl", val_lossacc_dir=DIR + "aep_val_lossacc.pkl",
                   reset_vars=False, chkpts_path=DIR+"aep_vars.pkl",
                   _load_model=True, _save_model=True)

aev_losses = train(ae_v, V_PATH, to_train="vel", files_
                    num_epochs=25, min_batch_size=128, 
                    trainlogdir = DIR + "aev_trainlog.txt", vallogdir = DIR + "aev_vallog.txt",
                    train_lossacc_dir=DIR + "aev_train_lossacc.pkl", val_lossacc_dir=DIR +"aev_val_lossacc.pkl",
                    reset_vars=True, chkpts_path=DIR+"aev_vars.pkl",
                    _load_model=False, _save_model=True)
