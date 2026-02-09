import os 
from datetime import datetime

import numpy as np
import torch 
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.onsetsandframes.evaluate import evaluate
from models.onsetsandframes.of import OnsetsAndFrames
from preprocessing.dataset import MAESTRO
from models.utils.constants import DEVICE
from models.onsetsandframes.utils import summary, cycle
from preprocessing.constants import MAX_MIDI, MIN_MIDI, N_MELS, SEQUENCE_LENGTH, DATA_PATH

ITERATIONS = 500000
CHECKPOINT_INTERVAL = 1000

BATCH_SIZE = 8
MODEL_COMPLEXITY = 48
if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
    BATCH_SIZE //= 2
    SEQUENCE_LENGTH //= 2
    print(f'Reducing batch size to {BATCH_SIZE} and sequence_length to {SEQUENCE_LENGTH} to save memory')

LEARNING_RATE = 0.0006
LEARNING_RATE_DECAY_STEPS = 10000
LEARNING_RATE_DECAY_RATE = 0.98

LEAVE_ONE_OUT = None

CLIP_GRADIENT_NORM = 3

VALIDATION_LENGTH = SEQUENCE_LENGTH
VALIDATION_INTERVAL = 500

logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')

def train():
    RESUME_ITERATION = None
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if LEAVE_ONE_OUT is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(LEAVE_ONE_OUT)})
        validation_groups = [str(LEAVE_ONE_OUT)]

    dataset = MAESTRO(path=DATA_PATH, groups=train_groups, sequence_length=SEQUENCE_LENGTH)
    validation_dataset = MAESTRO(path=DATA_PATH,groups=validation_groups, sequence_length=SEQUENCE_LENGTH)

    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)

    if RESUME_ITERATION is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, MODEL_COMPLEXITY).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
        RESUME_ITERATION = 0
    else:
        model_path = os.path.join(logdir, f'model-{RESUME_ITERATION}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))
    
    # summary(model)
    scheduler = StepLR(optimizer, step_size=LEARNING_RATE_DECAY_STEPS, gamma=LEARNING_RATE_DECAY_RATE)

    loop = tqdm(range(RESUME_ITERATION + 1, ITERATIONS + 1))
    for i, batch in zip(loop, cycle(loader)):
        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if CLIP_GRADIENT_NORM:
            clip_grad_norm_(model.parameters(), CLIP_GRADIENT_NORM)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % VALIDATION_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for key, value in evaluate(validation_dataset, model).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            model.train()

        if i % VALIDATION_INTERVAL == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

if __name__ == '__main__':
    train()
