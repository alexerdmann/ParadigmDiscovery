import os
# Sub libraries
from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib


class Inflector:

    def __init__(self, work_dir, args=None, data_format=dataloader.DataFormat.INFLECTION, model=model_lib.ModelFormat.TRANSFORMER, max_num_epochs=100, patience=7, batch_size=64, val_batch_size=1000, optimizer='adam', epsilon=0.000000001, num_layers=4, d_model=128, num_heads=8, dff=512, dropout_rate=0.1, beta_1=0.9, beta_2=0.98, warmup_steps=4000, extrinsic=False):

        self.work_dir = work_dir
        self.train = None
        self.dev = None
        self.test = None
        self.dev_acc = 0.0
        self.checkpoint_to_restore = None
        self.trained_model = None

        if args != None:
            self.data_format = args.s2s_data_format
            self.model = args.s2s_model
            self.max_num_epochs = args.s2s_max_num_epochs
            self.patience = args.s2s_patience
            self.batch_size = args.s2s_batch_size
            self.val_batch_size = args.s2s_val_batch_size
            self.optimizer = args.s2s_optimizer
            self.epsilon = args.s2s_epsilon
            self.num_layers = args.s2s_num_layers
            self.d_model = args.s2s_d_model
            self.num_heads = args.s2s_num_heads
            self.dff = args.s2s_dff
            self.dropout_rate = args.s2s_dropout_rate
            self.beta_1 = args.s2s_beta_1
            self.beta_2 = args.s2s_beta_2
            self.warmup_steps = args.s2s_warmup_steps
        else:
            self.data_format = data_format
            self.model = model
            self.max_num_epochs = max_num_epochs
            self.patience = patience
            self.batch_size = batch_size
            self.val_batch_size = val_batch_size
            self.optimizer = optimizer
            self.epsilon = epsilon
            self.num_layers = num_layers
            self.d_model = d_model
            self.num_heads = num_heads
            self.dff = dff
            self.dropout_rate = dropout_rate
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.warmup_steps = warmup_steps

        if extrinsic:  # We might have considered reducing the size of the model during EM for efficiency, but the final extrinsic model should be trained with full hyperparameters.
            self.max_num_epochs = 100
            self.patience = 40
            self.num_layers = 4
            self.d_model = 128
            self.dff = 512
        
    def prepare_s2s(self, train, dev=None, test=None):

        # Prepare to start training
        self.start_epoch = 0

        fn = os.path.join(os.path.dirname(self.work_dir), '{}_train.txt'.format(os.path.basename(self.work_dir)))
        with open(fn, 'w') as out_file:
            write_out_data_by_step(train, out_file)
        self.train = fn

        if dev:
            fn = os.path.join(os.path.dirname(self.work_dir), '{}_dev.txt'.format(os.path.basename(self.work_dir)))
            with open(fn, 'w') as out_file:
                write_out_data_by_step(dev, out_file)
            self.dev = fn

        if test:
            fn = os.path.join(os.path.dirname(self.work_dir), '{}_test.txt'.format(os.path.basename(self.work_dir)))
            with open(fn, 'w') as out_file:
                write_out_data_by_step(test, out_file)
            self.test = fn

    def continue_s2s(self, best_checkpoint_path, train, dev=None, test=None, warmup_steps=4000):

        self.prepare_s2s(train, dev=dev, test=test)
        self.checkpoint_to_restore = best_checkpoint_path
        self.warmup_steps = warmup_steps

    def train_validate_s2s(self):

        trained_model = seq2seq_runner.run(self, mode='ANA')
        self.dev_acc = trained_model.dev_acc
        self.trained_model = trained_model

        return trained_model.base_wf_tags_2_loss

    def train_validate_s2s_extrinsic(self):

        trained_model = seq2seq_runner.run(self)
        self.dev_acc = trained_model.dev_acc
        self.test_acc = trained_model.test_acc
        self.trained_model = trained_model

        return trained_model

def write_out_data_by_step(dataset, out_file):
    for (lem, wf, tup) in dataset:

        cluster, IC, context_vector_idx = tup
        printline = '{}\t{}\t{}'.format(lem, wf, cluster)
        if IC != None:
            printline += '\t{}'.format(IC)
        if context_vector_idx != None:
            printline += '\t{}'.format(context_vector_idx)

        out_file.write('{}\n'.format(printline))






