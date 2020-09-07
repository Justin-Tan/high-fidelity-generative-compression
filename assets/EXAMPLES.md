The images below were generated using the `HIFIC-low` model. You can generate the reconstructions by downloading the model weights and using the following command. The bitrate is reported in bits-per-pixel (`bpp`) of the compressed representation.

```bash
python3 compress.py -i /path/to/input/images --ckpt /path/to/model/checkpoint --reconstruct
```

Images are losslessly saved to PNG for viewing. More examples can be found in [this shared drive.](https://drive.google.com/drive/folders/1lH1pTmekC1jL-gPi1fhEDuyjhfe5x6WG).

Original | Reconstruction
:-------------------------:|:-------------------------:
![guess](assets/originals/cathedral_9.93bpp) | ![guess](assets/hific/cathedral_RECON_0.090bpp)

```python
Original: 9.93 bpp | HIFIC: 0.090 bpp
```


Original | Reconstruction
:-------------------------:|:-------------------------:
![guess](assets/originals/satellite_4.31bpp.png) | ![guess](assets/hific/satellite_RECON_0.039bpp.png)

```python
Original: 4.31 bpp | HIFIC: 0.039 bpp
```

Original | Reconstruction
:-------------------------:|:-------------------------:
![guess](assets/originals/telephone_5.61bpp.png) | ![guess](assets/hific/telephone_RECON_0.083bpp.png)

```python
Original: 5.61 bpp | HIFIC: 0.083 bpp
```

Original | Reconstruction
:-------------------------:|:-------------------------:
![guess](assets/originals/clocktower_9.93bpp.png) | ![guess](assets/hific/clocktower_RECON_0.090bpp.png)

```python
Original: 9.93 bpp | HIFIC: 0.090 bpp
```
