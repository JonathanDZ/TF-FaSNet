import torch
import torch.nn as nn
import copy

from utility.separator import Separator, SeparatorLayer, RNNModule, PositionalMultiHeadAttention


class EncoderSeparatorDecoder(nn.Module):
    """
    A standard Speech Separation architecture
    """

    def __init__(self, encoder, decoder, separator):
        super(EncoderSeparatorDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.separator = separator
    
    def forward(self, mix_audio):
        """
        Input: 
            mix_audio: Batch, nmic, T
        Output:
            separated_audio: Batch, nspk, T
        """
        enc_output, XrMM, ref_enc = self.encode(mix_audio)
        sep_output = self.separate(enc_output)
        dec_output = self.decode(sep_output, XrMM, ref_enc)
        return dec_output
    
    def encode(self, mix_audio):
        """
        Beside returning enc_output and ref_enc, encoder also returns a XrMM
        (Xr_magnitude_mean: mean of the magnitude of the ref channel of X)
        to ensure decoder properly perform inverse normalization.
        """
        return self.encoder(mix_audio)
    
    def separate(self, enc_output):
        return self.separator(enc_output)
    
    def decode(self, sep_output, XrMM, ref_enc):
        """
        Decoder receives XrMM to perform inverse normalization.
        """
        return self.decoder(sep_output, XrMM, ref_enc)


# Encoder
class Encoder(nn.Module):
    """
    STFT -> Normalization |-> view as Narrow-band -> BLSTM  |-> concat -> conv2d -> gLN
                          |-> take ref channel    -> conv2d |
    """

    def __init__(self, n_fft=256, embed_dim=32, nmic=6, dim_nb=64):
        """
        The sampling rate is 16 kHz. The STFT window size and hop size are 16 ms and 4 ms, respectively.
        n_fft = 16*16 = 256, hop_length = 4*16 = 64
        """
        super(Encoder, self).__init__()
        self.n_fft = n_fft
        self.window = torch.hann_window(n_fft)
        F = n_fft//2 + 1

        # For ref encode
        self.conv2d_1 = nn.Conv2d(in_channels=2, out_channels=embed_dim, kernel_size=3, padding=1)

        # For narrow-band encode
        # dim_nb = 2*embed_dim
        self.embed_dim = embed_dim
        self.conv1d = nn.Conv1d(nmic*2, dim_nb, 4)
        self.layernorm = nn.LayerNorm(dim_nb)
        self.rnn1 = nn.LSTM(input_size=dim_nb, hidden_size=dim_nb, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(2*dim_nb, dim_nb)
        self.deconv1d = nn.ConvTranspose1d(dim_nb, embed_dim, 4)

        # Final gLN
        self.conv2d_2 = nn.Conv2d(in_channels=2*embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.gLN = nn.GroupNorm(1, embed_dim, eps=1e-8)

    def forward(self, mix_audio):
        """
        mix_audio: batch, nmic, T
        """
        batch, nmic, T = mix_audio.shape
        output = mix_audio.view(-1, T) # batch*nmic, T
        output = torch.stft(output, self.n_fft,
                            window=self.window,
                            return_complex=True) # batch*nmic, n_fft/2 + 1, T'
        output = output.view(batch, nmic, output.shape[-2], output.shape[-1]) # batch, nmic, n_fft/2 + 1, T'
        output = output.permute(0,2,3,1).contiguous() # batch, n_fft/2 + 1, T', nmic

        # Normalization by using reference channel
        F, TF = output.shape[1], output.shape[2]
        ref_channel = 0
        Xr = output[... , ref_channel].clone() # Take a ref channel copy
        XrMM = torch.abs(Xr).mean(dim=2) # Xr_magnitude_mean: mean of the magnitude of the ref channel of X
        output[:, :, :, :] /= (XrMM.reshape(batch, F, 1, 1) + 1e-8)

        # View as real
        output = torch.view_as_real(output) # [B, F, TF, C, 2]

        # 1) Get ref channel encoding
        ref_enc = output[:,:,:,ref_channel,:].clone().permute(0,3,2,1).contiguous() # batch, 2, TF, n_fft/2 + 1
        ref_enc = self.conv2d_1(ref_enc) # batch, embed_dim, TF, n_fft/2 + 1

        # 2）Get all channel narrow-band encoding
        nb_enc_input = output.view(batch*F, TF, nmic*2).transpose(1,2).contiguous() # batch*n_fft/2+1, nmic*2, TF
        nb_enc_input = self.conv1d(nb_enc_input) # batch*nfft/2+1, embed_dim, TF
        nb_enc = nb_enc_input.transpose(1,2).contiguous() # batch*nfft/2+1, TF, embed_dim
        nb_enc = self.layernorm(nb_enc)

        nb_enc, _ = self.rnn1(nb_enc) # batch*nfft/2+1, TF, embed_dim*2
        nb_enc = self.linear1(nb_enc) # batch*nfft/2+1, TF, embed_dim

        nb_enc = nb_enc.transpose(1,2).contiguous() # batch*nfft/2+1, embed_dim, TF
        nb_enc = nb_enc + nb_enc_input
        nb_enc = self.deconv1d(nb_enc) # batch*nfft/2+1, embed_dim, TF
        nb_enc = nb_enc.view(batch, F, self.embed_dim, TF).permute(0,2,3,1).contiguous() # batch, embed_dim, TF, n_fft/2+1

        # 3) Concat two encodings to get a ifasnet-like encoding
        all_enc = torch.cat([ref_enc, nb_enc], 1) # batch, 2*embed_dim, TF, n_fft/2+1
        all_enc = self.conv2d_2(all_enc)
        all_enc = self.gLN(all_enc)

        return all_enc, XrMM, ref_enc


# Decoder
class Decoder(nn.Module):
    "Deconv2d -> linear -> view as full-band -> inverse normalization -> iSTFT"

    def __init__(self, n_fft=256, embed_dim=32, nspk=2, nmic=6, n_conv_layers=2, dropout=0.1, dim_ffn=128):
        super(Decoder, self).__init__()
        # For context decoding
        # dim_ffn = 4*embed_dim
        self.conv2d_in = nn.Conv2d(in_channels=2*embed_dim, out_channels=2*embed_dim, kernel_size=3, padding=1)
        self.gLN = nn.GroupNorm(1, 2*embed_dim, eps=1e-8)
        self.linear1 = nn.Linear(2*embed_dim, dim_ffn)
        self.activation = nn.functional.silu

        convs = []
        for l in range(n_conv_layers):
            convs.append(nn.Conv2d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=3, padding='same', groups=dim_ffn, bias=True))
            convs.append(nn.GroupNorm(4, dim_ffn, eps=1e-8))
            convs.append(nn.SiLU())
        self.conv = nn.Sequential(*convs)

        # self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffn, 2*embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Decode
        self.nspk = nspk
        self.deconv2d = nn.ConvTranspose2d(in_channels=2*embed_dim, out_channels=2*nspk, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=2*nspk, out_features=2*nspk)
        self.n_fft = n_fft
        self.window = torch.hann_window(n_fft)
    
    def forward(self, x, XrMM, ref_enc):
        """
        x: batch, D, T, F
        ref_enc: batch, D, T, F
        """
        batch, _, T, F = x.shape

        # Add a decode process, which utilizes ref_enc info to potentially enhance its performance
        embedding_input = torch.cat([ref_enc, x], 1) # batch, 2*D, T, F
        embedding_input = self.conv2d_in(embedding_input) # batch, 2*D, T, F
        embedding = self.gLN(embedding_input) # batch, 2*D, T, F
        embedding = self._ff_block(embedding) # batch, 2*D, T, F
        embedding = embedding + embedding_input

        output = self.deconv2d(embedding) # batch, 2*nspk, T, F
        output = output.permute(0,3,2,1).contiguous() # batch, F, T, 2*nspk
        output = self.linear(output)

        # To complex
        output = output.view(batch, F, T, self.nspk, 2) # batch, F, T, nspk, 2
        output = torch.view_as_complex(output) # batch, F, T, nspk

        # Inverse normalization
        Ys_hat = torch.empty(size=(batch, self.nspk, F, T), dtype=torch.complex64, device=output.device)
        XrMM = torch.unsqueeze(XrMM, dim=2).expand(-1, -1, T)
        for spk in range(self.nspk):
            Ys_hat[:, spk, :, :] = output[:, :, :, spk] * XrMM[:, :, :]

        # iSTFT with frequency binding
        ys_hat = torch.istft(Ys_hat.view(batch * self.nspk, F, T), n_fft=self.n_fft, window=self.window, win_length=self.n_fft)
        ys_hat = ys_hat.view(batch, self.nspk, -1)
        return ys_hat
    
    # Feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        "x: B, 2*D, T, F"
        x = x.transpose(1,3).contiguous() # B, F, T, 2*D
        x = self.linear1(x) # B, F, T, 2*D
        x = self.activation(x)
        x = x.transpose(1,3).contiguous() # B, 2*D, T, F
        x = self.conv(x)
        x = x.transpose(1,3).contiguous() # B, F, T, 2*D
        # x = self.dropout1(x)
        x = self.linear2(x)
        x = x.transpose(1,3).contiguous() # B, 2*D, T, F 
        return self.dropout2(x)
    

def make_TF_FaSNet(nmic=6, nspk=2, n_fft=256, embed_dim=16, dim_nb=32, dim_ffn=64, n_conv_layers=2, B=4, I=8, J=1, H=128, L=4):
    "Helper: Construct TF-FaSNet model from hyperparameters"
    F = n_fft//2 + 1
    E = embed_dim//L

    c = copy.deepcopy
    RNN_module = RNNModule(hidden_size=H, kernel_size=I, stride=J, embed_dim=embed_dim)
    self_attn = PositionalMultiHeadAttention(h=L, d_model=embed_dim, d_q=E, F=F)
    model = EncoderSeparatorDecoder(
        encoder=Encoder(n_fft=n_fft, embed_dim=embed_dim, nmic=nmic, dim_nb=dim_nb), 
        decoder=Decoder(n_fft=n_fft, embed_dim=embed_dim, nspk=nspk, n_conv_layers=n_conv_layers, dim_ffn=dim_ffn),
        separator=Separator(SeparatorLayer(c(RNN_module), c(RNN_module), c(self_attn)), N=B)
    )

    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6

if __name__ == "__main__":

    test_model = make_TF_FaSNet()
    test_model.eval()

    # Check model size
    print(check_parameters(test_model))

    # Test full model
    mix_audio = torch.randn(3,6,64000)
    separated_audio = test_model(mix_audio)
    print(separated_audio.shape)
