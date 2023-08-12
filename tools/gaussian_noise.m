function[noise] = gaussian_noise(r, c, a, b)

noise = randn(r, c);

noise = noise - mean(noise);
noise = noise - std(noise);

noise = a + b*noise;

end