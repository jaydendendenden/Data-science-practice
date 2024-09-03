from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from data_loader import load_data
def load():
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()

    return graph, features, train_labels, train_mask, test_mask