from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data_loader import DataLoader
from data_gen import DataGen
import time
import os.path


def create_log_dirs(dirs, data_path):
    for dir in dirs:
        if not os.path.exists("{}{}".format(data_path, dir)):
            os.mkdir("{}{}".format(data_path, dir))

# run tb -> tensorboard --logdir=logs/tensorboard/ --host localhost --port 8088
def train(model, data_path, sequence_length=30, batch_size=32, nb_epoch=100, split_strat=1, split=0.3):

    create_log_dirs(["logs/", "logs/csv/", "logs/tensorboard", "logs/checkpoints"], data_path)
    model_name = "{}-{}".format(model, time.time())

    # Helper: Save the model.
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(data_path, 'logs', 'checkpoints', model_name + '-' + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir="{}logs/tensorboard/{}".format(data_path, model_name))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save data in csv
    csv_logger = CSVLogger(os.path.join(data_path, 'logs', 'csv', model_name + '-' + 'training-' + str(time.time()) + '.csv'))

    # Training model
    data_loader = DataLoader(data_path, "frames", split_strat=split_strat, split=split)

    if model in ["lstm"]:
        data_loader = DataLoader(data_path, "features", split_strat=split_strat, split=split)

    X, y, X_test, y_test, n_classes = data_loader.load_data()
    rm = ResearchModels(n_classes, model, sequence_length)
    print(X.shape)

    rm.model.fit(X, y,batch_size=batch_size,validation_data=(X_test, y_test),
                 verbose=1,
                 callbacks=[tb, early_stopper, csv_logger, checkpoint],
                 epochs=nb_epoch)


#model = DataGen("hmdb/", fpv=24)
#model.generate_data()


train("lstm", "hmdb_op_24/", sequence_length=24, split_strat=2)
# train("lstm", "nada_op_30/", split_strat=1,split=0.36)