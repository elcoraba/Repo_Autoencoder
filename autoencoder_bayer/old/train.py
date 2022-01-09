## in WORK
import numpy as np
import torch.nn as nn 
import torch
import pickle

def save_checkpoints(path: str, *args):
  with open(path, 'wb') as f:                       #with: It allows you to ensure that a resource is "cleaned up" when the code that uses it finishes running, even if exceptions are thrown
    pickle.dump(args, f)                            #Write the pickled representation of the object obj to the open file object file.

def load_checkpoints(path: str):
  with open(path, 'rb') as f:
    return pickle.load(f)                           #Read the pickled representation of an object from the open file object file and return the reconstituted object hierarchy specified therein.
  
def load_model(model: nn.Module, path: str, device: str ='cpu'):
  model.load_state_dict(torch.load(path, map_location=device))
  return model

def save_model(model: nn.Module, path: str):
  torch.save(model.state_dict(), path)

###############################################################################################################################################
def train(ae, ae_path, to_train, files_, num_epochs = 14, min_batch_size=256, BATCH_SIZE=10,  lr=5e-4, _train_len=640, _test_len=160, 
          reset_vars=False, chkpts_path="vars.pkl", vallogdir=None, trainlogdir=None, train_lossacc_dir=None, val_lossacc_dir=None,
          _load_model=False, _save_model=False):
  assert to_train in ["vel", "pos"]
  
  print("Total files", len(files_))
    
  if _load_model:
    ae = load_model(ae, ae_path, device)
  ae.to(device)

  if reset_vars:
    save_checkpoints(chkpts_path, 0,0)
    save_checkpoints(train_lossacc_dir, [])
    save_checkpoints(val_lossacc_dir, [])
  
  # lade checkpoints
  train_lossacc = load_checkpoints(train_lossacc_dir)[0]        #train dataset
  val_lossacc = load_checkpoints(val_lossacc_dir)[0]            #validation dataset
  print("Loaded train_loss", train_lossacc)
  print("Loaded val_loss", val_lossacc)

  saved_epoch, saved_batchidx = load_checkpoints(chkpts_path)
  
  train_dataset, test_dataset = random_split(files_, [_train_len, _test_len], generator=torch.Generator().manual_seed(SEED))

  losses = []
  val_losses = []
  criterion = nn.MSELoss()                              
  optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
  
  for epoch in range(saved_epoch, num_epochs):
    for batch_idx in range(saved_batchidx, len(train_dataset), BATCH_SIZE):     # Anfang Ende Schrittgröße
      if int(saved_batchidx/BATCH_SIZE)+1 >= int(len(train_dataset)/BATCH_SIZE):# Sind dann mit einer Epoch fertig
        print(f"Saved_batch {int(saved_batchidx/BATCH_SIZE)+1} is greater or equal to total batches {int(len(train_dataset)/BATCH_SIZE)}")
        break
    
      current_batch = list(train_dataset)[batch_idx: batch_idx + BATCH_SIZE]
      controller.reset()                                                        #TODO ?
      print(f"Batch idx {batch_idx}, from {batch_idx} to {batch_idx+ BATCH_SIZE} of {len(list(train_dataset))}")
      print("Current batch:", current_batch)
      run(plugin_manager, current_batch)                                        # preprocessing von data (siehe weiter oben)

      progresses = ""

      pos_t = []
      vels_t = []

      for data in controller.model:                                         # hole position und velocity aus preprocessed data und packe sie zsm
        pos, vels = encode(data)                                            # auch preprocessing: normalisieren, alle invaliden daten weg alle NaNs weg... weiter oben
        pos_t.append(prepare_tensor(pos))
        vels_t.append(prepare_tensor(vels))
        
      pos_t = cat(pos_t)                                                    # zu einem großen tensor concatiniert
      vels_t = cat(vels_t)

      print("AE (positions) input (Expanded sample size):\t", pos_t.shape)
      print("AE (velocities) input (Expanded sample size):\t", vels_t.shape)

      if to_train is "pos":
        
        train_data = TensorDataset(pos_t)
      else:
        train_data = TensorDataset(vels_t)
      train_loader = DataLoader(dataset=train_data, batch_size=min_batch_size, shuffle=True, drop_last=True)    

      for min_batch_x in train_loader:
        ae.train()                                      #model wird nun in train zustand versetzt, also zB Dropout layer wird wirklich angewendet

        x_batch = min_batch_x[0].to(device)             #TODO virmarie [0]?
        
        out, z = ae(x_batch)

        loss = criterion(out, x_batch)
        loss.backward()
        losses.append(loss.data)

        optimizer.step()
        optimizer.zero_grad()

        #zB 1/10 epochs ....
        progress = f'epoch: [{epoch+1} / {num_epochs}], batch: [{int(batch_idx/BATCH_SIZE)+1} / {int(len(train_dataset)/BATCH_SIZE)}], loss: {loss.data} \n'
        train_lossacc.append([epoch, int(batch_idx/BATCH_SIZE)+1, int(len(train_dataset)/BATCH_SIZE), loss.data])

        print(progress)
        progresses += progress
      
      #training fertig 

      if trainlogdir is not None:
        with open(trainlogdir, 'a+') as log:
          log.write(progresses)                            # write progresses in unser file
        print("Saved train console log.")

      if train_lossacc_dir is not None:
        save_checkpoints(train_lossacc_dir, train_lossacc)
        print("Saved train losses.")
  
      save_checkpoints(chkpts_path, epoch, batch_idx)
      if _save_model:
        # torch.save(ae.state_dict(), ae_path)
        save_model(ae, ae_path)
        print("Saved model.")

    ### Validation              #das gleiche nochmal mit testdatenset - sollten in epoch schleife sein # pro epoch einmal alle testbatches durch, dann alle validation batches durch und dann neue epoch
    controller.reset()
    run(plugin_manager, test_dataset)

    pos_t = []
    vels_t = []

    for data in controller.model:
      pos, vels = encode(data)
      pos_t.append(prepare_tensor(pos))
      vels_t.append(prepare_tensor(vels))
        
    pos_t = cat(pos_t)
    vels_t = cat(vels_t)

    if to_train is "pos":
      test_data = TensorDataset(pos_t)
    else:
      test_data = TensorDataset(vels_t)
    
    test_loader = DataLoader(dataset=test_data, batch_size=min_batch_size, shuffle=True, drop_last=True)

    with torch.no_grad():                           #gradienten werden nicht erzeugt/gespeichert
      progresses = ""
      for val_batch, x_val in enumerate(test_loader): #idx, wert
        x_val = x_val[0].to(device)
        
        ae.eval()
        
        out, z = ae(x_val)
        val_loss = criterion(out, x_val)
        val_losses.append(val_loss.data)

        progress = f'epoch: [{epoch+1} / {num_epochs}], batch: [{val_batch+1} / {len(test_loader)}], validation loss: {val_loss.data} \n'
        val_lossacc.append([epoch, val_batch+1, len(test_loader), val_loss.data])

        print(progress)
        progresses += progress
      
      if vallogdir is not None:
        with open(vallogdir, 'a+') as vallog:
          vallog.write(progresses)
        print("Saved val console log.")

      if val_lossacc_dir is not None:
        save_checkpoints(val_lossacc_dir, val_lossacc)
        print("Saved val losses.")
    
    save_checkpoints(chkpts_path, epoch+1, 0)
    saved_batchidx = 0

# [epoch, int(batch_idx/BATCH_SIZE)+1, int(len(train_dataset)/BATCH_SIZE), loss.data]
# [epoch, val_batch+1, len(test_loader), val_loss.data]
  return train_lossacc, val_lossacc           #TODO @virmarie val_lossacc_dir? müsste doch val_lossacc sein oder?