"""
Modelos de Deep Learning para clasificación de convulsiones en EEG
Compatible con skorch para integración con sklearn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EEGTransformer(nn.Module):
    """
    Transformer para clasificación de señales EEG
    Input: (batch_size, seq_len, input_dim)
    Output: (batch_size, num_classes)
    """
    def __init__(self, input_dim=19, seq_len=3000, d_model=64, nhead=4, 
                 num_layers=2, dropout=0.1, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.cls_head(x)   # (B, num_classes)
        return x


class PositionalEncoding(nn.Module):
    """Positional Encoding para Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNN1DClassifier(nn.Module):
    """
    CNN 1D para clasificación (trabaja con features extraídas)
    Input: (batch_size, input_features)
    Output: (batch_size, num_classes)
    """
    def __init__(self, input_features=50, conv1_channels=64, conv2_channels=128,
                 fc_hidden=64, dropout=0.5, num_classes=2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_channels, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=conv1_channels, out_channels=conv2_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calcular tamaño después de convoluciones
        self.fc1_input_features = self._get_conv_output_size(input_features, 
                                                              conv1_channels, 
                                                              conv2_channels)
        
        self.fc1 = nn.Linear(self.fc1_input_features, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        # x: (B, F) donde F = input_features
        x = x.unsqueeze(1)  # (B, 1, F)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _get_conv_output_size(self, input_length, conv1_ch, conv2_ch):
        """Calcula el tamaño de salida después de convoluciones"""
        dummy_input = torch.rand(1, 1, input_length)
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.view(1, -1).size(1)


class LSTMClassifier(nn.Module):
    """
    LSTM para clasificación de series temporales EEG
    Input: (batch_size, seq_len, input_dim)
    Output: (batch_size, num_classes)
    """
    def __init__(self, input_dim=19, hidden_dim=128, num_layers=2, 
                 dropout=0.3, bidirectional=True, num_classes=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Usar el último hidden state
        if self.lstm.bidirectional:
            # Concatenar forward y backward del último layer
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]
        
        output = self.classifier(hidden)
        return output


class GRUClassifier(nn.Module):
    """
    GRU para clasificación de series temporales EEG
    Input: (batch_size, seq_len, input_dim)
    Output: (batch_size, num_classes)
    """
    def __init__(self, input_dim=19, hidden_dim=128, num_layers=2,
                 dropout=0.3, bidirectional=True, num_classes=2):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        gru_out, h_n = self.gru(x)
        
        # Usar el último hidden state
        if self.gru.bidirectional:
            # Concatenar forward y backward del último layer
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]
        
        output = self.classifier(hidden)
        return output


# ============================================================
# Dataset personalizado para DL
# ============================================================

from torch.utils.data import Dataset

class EEGWindowDataset(Dataset):
    """
    Dataset para ventanas de EEG
    Para modelos que trabajan con datos raw (Transformer, LSTM, GRU)
    """
    def __init__(self, df, n_channels=19, seq_len=3000):
        self.data = []
        self.labels = []
        
        grouped = df.groupby("window_id")
        for window_id, group in grouped:
            # Columnas de canales EEG (ajustar según tu CSV)
            X = group.iloc[:, 4:4 + n_channels].values  # (T, C)
            y = group["Seizure"].values[0]
            
            if X.shape[0] == seq_len:  # Verificar longitud
                self.data.append(X)
                self.labels.append(y)
        
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class EEGFeaturesDataset(Dataset):
    """
    Dataset para features extraídas (TSFRESH)
    Para modelos que trabajan con features (CNN 1D sobre features)
    """
    def __init__(self, X, y):
        """
        Args:
            X: pandas DataFrame o numpy array de features
            y: pandas Series o numpy array de labels
        """
        if hasattr(X, 'values'):
            self.X = torch.tensor(X.values, dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
            
        if hasattr(y, 'values'):
            self.y = torch.tensor(y.values, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
