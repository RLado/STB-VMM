from PIL import Image
import pad_img
import argparse


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Add black borders to an image to achieve a resolution divisible by d')
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Datasets parameters
    required.add_argument('-i', '--input', type=str, help='Input image', required=True)
    required.add_argument('-d', '--d', type=int, help='Resolution must be divisible by "d"', required=True)

    optional.add_argument('-f', '--fill_colour', type=int, nargs=3 ,default=(0, 0, 0),
                          help='Border colour')
    optional.add_argument('-o', '--output', type=str, default=None, help='Output path')
    args = parser.parse_args()

    #Pad
    img = Image.open(args.input)
    img = pad_img.auto_pad(img, args.d, args.fill_colour)
    if args.output == None:
        img.show()
    else:
        img.save(args.output)
