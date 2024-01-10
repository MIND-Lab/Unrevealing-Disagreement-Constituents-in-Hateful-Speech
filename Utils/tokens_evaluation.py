import matplotlib as mpl

def get_all_colors(tokens_list, tokens_df):
  """
  Retrieves color information for each token in the given list.
  For a given sentence create a vector with the tokens score:

  Args:
      tokens_list (list): A list of tokens (words) representing a sentence.

  Returns:
      tuple: A tuple containing two lists:
          - colors_agreement: A list of color information corresponding to the agreement coordinates
            of each token. If a token is not found in the token DataFrame, 'NA' is appended.
          - colors_hate: A list of color information corresponding to the hate coordinates
            of each token. Note: This list is currently commented out in the code.
  """
  colors_agreement = []  # List to store color information for agreement coordinates
  colors_hate = []  # List to store color information for hate coordinates

  for token in tokens_list:
      if token.lower() not in list(tokens_df['token']):
          colors_agreement.append('NA')  # Token not found in token DataFrame, 'NA' is appended
      else:
          # Token found in token DataFrame, append its agreement coordinate color information
          colors_agreement.append(tokens_df.loc[tokens_df['token'] == token.lower(),
                                                  'Agreement_coordinate'].values[0])

      # Note: The following line is currently commented out in the code
      # colors_hate.append(tokens_df.loc[tokens_df['token'] == token.lower(), 'Hate_coordinate'].values[0])

  return colors_agreement, colors_hate

def find_NA_indices(list_to_check):
    """
    Finds the indices of 'NA' values in a list.

    Args:
        list_to_check (list): List to check for 'NA' values.

    Returns:
        list: List of indices where 'NA' values are found.

    """
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == 'NA':
            indices.append(idx)
    return indices



def colorize(attrs, cmap='PiYG'):
    """
    Colorizes a list of attributes using a specified colormap.

    Args:
        attrs (list): List of attributes.
        cmap (str, optional): Colormap name. Defaults to 'PiYG'.

    Returns:
        list: List of colors in hexadecimal format.

    """

    indexes = []

    if 'NA' in attrs:
        # Find indices of 'NA' values in attrs
        indexes = find_NA_indices(attrs)

        # Replace 'NA' values with 0
        for i in indexes:
            attrs[i] = 0

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmap = mpl.cm.get_cmap(cmap)

    # Convert attribute values to colors using the colormap
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))

    if indexes:
        # Set colors for 'NA' values to gray (#A9A9A9)
        for i in indexes:
            colors[i] = '#A9A9A9'  # '#FFFF00'

    return colors