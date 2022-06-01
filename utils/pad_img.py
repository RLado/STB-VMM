import argparse
from PIL import Image


def pad_img(img, tw: int, th: int, fill_colour=(0, 0, 0)):
    '''
    Adds black borders to an image. (This function does not downscale)

    Parameters:
        img (PIL.Image): Input image to pad
        tw (int): New width size
        th (int): New height size
        fill_colour (tuple): Color instead of black (optional)
    
    Returns:
        PIL.Image: Resulting padded image

    '''

    # Check size
    x, y = img.size
    if tw < x:
        tw = x
    if th < y:
        th = y

    # Generate background and paste image 
    new_img = Image.new('RGB', (tw, th), fill_colour)
    new_img.paste(img, ((tw-x) // 2, (th-y) // 2))

    return new_img

def auto_pad(img, d=64, fill_colour=(0, 0, 0)):
    '''
    Pad image to make it divisible by "d".

    Parameters:
        img (PIL.Image): Input image to pad
        d (int): Pad to be divisible by d (defaults to 64)
        fill_colour (tuple): Color instead of black (optional)
    
    Returns:
        PIL:Image: Resulting padded image
    
    '''

    x, y = img.size
    if x%d != 0:
        x = (x//d+1)*d
    if y%d != 0:
        y = (y//d+1)*d
    return pad_img(img, x, y, fill_colour)

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Add black borders to an image')
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Datasets parameters
    required.add_argument('-i', '--input', type=str, help='Input image', required=True)
    required.add_argument('-t', '--target', type=int, nargs=2, help='Target size w h', required=True)

    optional.add_argument('-f', '--fill_colour', type=int, nargs=3 ,default=(0, 0, 0),
                          help='Border colour')
    optional.add_argument('-o', '--output', type=str, default=None, help='Output path')
    args = parser.parse_args()

    #Pad
    img = Image.open(args.input)
    img = pad_img(img, args.target[0], args.target[1], args.fill_colour)
    if args.output == None:
        img.show()
    else:
        img.save(args.output)