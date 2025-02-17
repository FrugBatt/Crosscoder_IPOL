import torch
import unittest
from crosscoder import Encoder, CrossCoderDecoder, CrossCoder

class TestShapes(unittest.TestCase):

    def test_encoder_shape(self):
        activation_dim = 64
        dict_size = 128
        num_layers = 4
        encoder = Encoder(activation_dim, dict_size, num_layers)
        x = torch.randn(10, num_layers, activation_dim) # Example input
        f = encoder(x)
        self.assertEqual(f.shape, (10, dict_size))

    def test_decoder_shape(self):
        activation_dim = 64
        dict_size = 128
        num_layers = 4
        decoder = CrossCoderDecoder(activation_dim, dict_size, num_layers)
        f = torch.randn(10, dict_size)  # Example input
        x = decoder(f)
        self.assertEqual(x.shape, (10, num_layers, activation_dim))

    def test_crosscoder_shape(self):
        activation_dim = 64
        dict_size = 128
        num_layers = 4
        crosscoder = CrossCoder(activation_dim, dict_size, num_layers)
        x = torch.randn(10, num_layers, activation_dim) # Example input
        x_hat = crosscoder(x)
        self.assertEqual(x_hat.shape, (10, num_layers, activation_dim))

    def test_crosscoder_output_features_shape(self):
      activation_dim = 64
      dict_size = 128
      num_layers = 4
      crosscoder = CrossCoder(activation_dim, dict_size, num_layers)
      x = torch.randn(10, num_layers, activation_dim)
      x_hat, f_scaled = crosscoder(x, output_features=True)
      self.assertEqual(x_hat.shape, (10, num_layers, activation_dim))
      self.assertEqual(f_scaled.shape, (10, 128))

    def test_get_activations_shape(self):
      activation_dim = 64
      dict_size = 128
      num_layers = 4
      crosscoder = CrossCoder(activation_dim, dict_size, num_layers)
      x = torch.randn(10, num_layers, activation_dim)
      activations = crosscoder.get_activations(x)
      self.assertEqual(activations.shape, (10, 128))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
