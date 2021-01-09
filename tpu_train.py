import numpy
import os
import time
import logging

import torch_xla
import torch_xla.core.xla_model as xm
import torch.optim as optim

from models import DPTNet_base
from others.data import DummyDataset, DummyDataLoader
from others.pit_criterion import cal_loss

def main():
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger('pytorch')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(os.environ['HOME'],
        '{}-{}.log'.format('tpu', time.strftime("%Y%m%d-%H%M%S"))))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('Logger setup')
    
    device = xm.xla_device()
    
    tr_dataset = DummyDataset(sample_rate=8000, segment=4.0)
    cv_dataset = DummyDataset(data_len=10, sample_rate=8000)
    tr_loader = DummyDataLoader(tr_dataset, batch_size=1,
                                shuffle=True)
    cv_loader = DummyDataLoader(cv_dataset, batch_size=1, shuffle=False)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    logger.info('Data ready')
    
    model = DPTNet_base(enc_dim=64, feature_dim=64, hidden_dim=128, layer=2, segment_size=256, nspk=2, win_len=16)
    model = model.to(device)
    logger.info('Model on device')
    
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    
    tracker = xm.RateTracker()
    
    epoch = 0
    total_loss = 0
    start = time.time()
    model.train()
    logger.info('Starting train')
    
    for i, (data) in enumerate(tr_loader):
        padded_mixture_, mixture_lengths_, padded_source_ = data
        seg_idx = numpy.random.randint(0, padded_mixture_.shape[0], 1)
        padded_mixture = padded_mixture_[seg_idx, :]
        mixture_lengths = mixture_lengths_[seg_idx]
        padded_source = padded_source_[seg_idx, :]
        if True:
            padded_mixture = padded_mixture.to(device)
            mixture_lengths = mixture_lengths.to(device)
            padded_source = padded_source.to(device)
    
        logger.info('Got data')
        estimate_source = model(padded_mixture)
        logger.info('Got output')
        loss, max_snr, estimate_source, reorder_estimate_source = \
            cal_loss(padded_source, estimate_source, mixture_lengths)
        logger.info('Got loss')
        if True:
    
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            xm.optimizer_step(optimizer)
            tracker.add(1)
            
        if i % 100 == 0:
            logger.info('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Current Loss {3:.6f} | Rate {4} | Glob Rate {5}'.format(
                epoch + 1, i + 1, total_loss / (i + 1),
                loss.item(), tracker.rate(),
                tracker.global_rate(),
                flush=True))
    
    logger.info('Done')


if __name__ == '__main__':
    main()