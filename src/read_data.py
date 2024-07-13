"""
Read Hyperspectral Cube data from Bayspec GoldenEye
"""

import numpy as np
from pathlib import Path
import cv2

DEFAULT_WAVELENGTHS = np.array(
    [
        400.87586752,
        405.95512693,
        411.03004636,
        416.10066439,
        421.16701956,
        426.22915045,
        431.2870956,
        436.34089359,
        441.39058296,
        446.43620229,
        451.47779013,
        456.51538504,
        461.54902558,
        466.57875032,
        471.60459781,
        476.62660661,
        481.64481529,
        486.65926241,
        491.66998652,
        496.67702619,
        501.68041997,
        506.68020643,
        511.67642413,
        516.66911163,
        521.65830749,
        526.64405026,
        531.62637852,
        536.60533082,
        541.58094571,
        546.55326177,
        551.52231755,
        556.48815162,
        561.45080252,
        566.41030883,
        571.3667091,
        576.3200419,
        581.27034578,
        586.21765931,
        591.16202104,
        596.10346954,
        601.04204337,
        605.97778108,
        610.91072124,
        615.84090241,
        620.76836315,
        625.69314202,
        630.61527758,
        635.53480839,
        640.45177301,
        645.36621,
        650.27815792,
        655.18765533,
        660.0947408,
        664.99945288,
        669.90183014,
        674.80191113,
        679.69973441,
        684.59533855,
        689.48876211,
        694.38004364,
        699.26922171,
        704.15633487,
        709.0414217,
        713.92452074,
        718.80567056,
        723.68490972,
        728.56227678,
        733.4378103,
        738.31154884,
        743.18353096,
        748.05379523,
        752.92238019,
        757.78932442,
        762.65466648,
        767.51844491,
        772.38069829,
        777.24146518,
        782.10078413,
        786.95869371,
        791.81523247,
        796.67043898,
        801.5243518,
        806.37700949,
        811.2284506,
        816.0787137,
        820.92783735,
        825.77586012,
        830.62282055,
        835.46875721,
        840.31370867,
        845.15771347,
        850.00081019,
        854.84303738,
        859.68443361,
        864.52503743,
        869.3648874,
        874.20402209,
        879.04248005,
        883.88029985,
        888.71752005,
        893.5541792,
        898.39031587,
        903.22596862,
        908.061176,
        912.89597659,
        917.73040893,
        922.5645116,
        927.39832314,
        932.23188213,
        937.06522711,
        941.89839666,
        946.73142933,
        951.56436368,
        956.39723828,
        961.23009168,
        966.06296245,
        970.89588914,
        975.72891031,
        980.56206453,
        985.39539036,
        990.22892636,
        995.06271108,
        999.89678309,
        1004.73118095,
        1009.56594321,
        1014.40110845,
        1019.23671521,
        1024.07280207,
        1028.90940758,
        1033.7465703,
        1038.58432879,
        1043.42272161,
        1048.26178733,
        1053.10156449,
        1057.94209168,
        1062.78340743,
        1067.62555033,
        1072.46855891,
        1077.31247176,
        1082.15732742,
        1087.00316445,
    ]
)


def interpret_value(value):
    if "{" in value:
        try:
            return np.array(
                [float(chunk.strip()) for chunk in value.strip("{}").split(",")]
            )
        except ValueError:
            pass
    try:
        return int(value)
    except ValueError:
        return value


def read_header(fname: Path):
    header_info = {}
    with open(fname, "r") as f:
        assert "ENVI" == f.readline().strip()
        for line in f:
            if "=" in line:
                key, value = line.split("=")[0].strip(), line.split("=")[1].strip()
                if "{" in value:
                    while "}" not in value:
                        value += f.readline().strip()
                header_info[key] = interpret_value(value)
    with open(fname, "r") as f:
        assert len([1 for c in f.read() if c == "="]) == len(
            header_info
        ), "missed some header info"
    assert (
        len(header_info["wavelength"]) == header_info["bands"]
    ), "wavelengths not read properly"
    return header_info


def read_data(fname: Path, header_info=None):
    if header_info is None:
        if "ms" in fname.suffix:  # no file extension
            header_info = read_header(str(fname) + ".hdr")
        else:
            header_info = read_header(Path(fname).with_suffix(".hdr"))
    cols = header_info["samples"]
    rows = header_info["lines"]
    bands = header_info["bands"]

    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.uint16)
    core = data

    assert (
        core.shape[0] == bands * rows * cols
    ), f"file format error: {core.shape[0]}, {bands}, {rows}, {cols}, {fname}"
    img = core.reshape((bands, rows, cols)).transpose((1, 2, 0))  # (H, W, C)

    return np.flipud(img)


def hs_to_rgb(img, wavelengths=DEFAULT_WAVELENGTHS, white=None):
    rgb_wavelengths = [620, 555, 503]
    nearest = wavelengths.reshape(-1, 1) - np.array(rgb_wavelengths).reshape(1, -1)
    nearest = np.argmin(np.abs(nearest), axis=0)
    # return (img[:, :, nearest] >> 4)  # shift from 12-bit to 8-bit
    if white is None:
        return np.minimum((1 << 8) - 1, img[:, :, nearest])  # threshold to 8-bit
    if white == "default":
        white = [176.90352, 94.22424, 101.18808]
    return ((img[:, :, nearest] / white).clip(0, 1) * 255).astype(np.uint8)


def main(src: Path, dst: Path = None, rgb_only: bool = False):
    import tqdm

    if isinstance(src, str):
        src = Path(src)
    if isinstance(dst, str):
        dst = Path(dst)

    if dst is None:
        dst = src
    if not dst.exists():
        dst.mkdir(parents=True)
    if not rgb_only:
        for wavelength in DEFAULT_WAVELENGTHS:
            (dst / str(wavelength)).mkdir(exist_ok=True)

    outputs = []
    # main loop
    for file in tqdm.tqdm(list(src.glob("*.hdr"))):
        cubefilename = file.with_suffix("")
        if not cubefilename.exists():
            cubefilename = file.with_suffix(".bin")
        try:
            img = read_data(cubefilename)
            fname = file.with_suffix(".jpg").name
            print("img.shape", img.shape)
            cv2.imwrite(str(dst / fname), (hs_to_rgb(img)))
            if not rgb_only:
                for wavelength, gray in zip(
                    DEFAULT_WAVELENGTHS, img.transpose(2, 0, 1)
                ):
                    cv2.imwrite(str(dst / str(wavelength) / fname), (gray >> 4))
            outputs.append(img)
        except AssertionError as e:
            print(e)

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Read Hyperspectral Cubes",
        description="Reads Hyperspectral Cubes and outputs them into a more useable format",
    )
    parser.add_argument(
        "source", help="The source folder containing the hyperspectral cubes"
    )
    parser.add_argument(
        "dest",
        nargs="?",
        default=None,
        help="The destination folder to save the hyperspectral images",
    )
    parser.add_argument("--rgb-only", action="store_true")
    args = parser.parse_args()
    if args.dest is None:
        main(Path(args.source), rgb_only=args.rgb_only)
    else:
        main(Path(args.source), Path(args.dest), rgb_only=args.rgb_only)
