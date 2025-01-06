import numpy as np
import matplotlib.pyplot as plt

"""
Creates Pyramid pattern in a recurse manner
"""
def create_pattern(patterns, gap, random=False):
    pattern = patterns[0]
    new_pattern = np.zeros((2 * pattern.shape[0] + 7 * gap, 2 * pattern.shape[1] + 7 * gap))
    # print(new_pattern.shape)
    # place walls
    new_pattern[:gap,:] = 1
    new_pattern[:,:gap] = 1
    new_pattern[-gap:,:] = 1
    new_pattern[:,-gap:] = 1

    # place interior walls
    new_pattern[3 * gap + pattern.shape[0]:4 * gap + pattern.shape[0],:] = 1
    new_pattern[:,3 * gap + pattern.shape[1]:4 * gap + pattern.shape[1]] = 1

    # place smaller pattern in each new room
    new_pattern[(2 * gap):(2 * gap + pattern.shape[0]), (2 * gap):(2 * gap + pattern.shape[1])] = patterns[0]
    new_pattern[(5 * gap + pattern.shape[0]):(5 * gap + 2 * pattern.shape[0]), (2 * gap):(2 * gap + pattern.shape[1])] = patterns[1]

    new_pattern[(5 * gap + pattern.shape[0]):(5 * gap + 2 * pattern.shape[0]), (5 * gap + pattern.shape[1]):(5 * gap + 2 * pattern.shape[1])] = patterns[2]
    new_pattern[(2 * gap):(2 * gap + pattern.shape[0]), (5 * gap + pattern.shape[1]):(5 * gap + 2 * pattern.shape[1])] = patterns[3]

    # place a gap in one of the corners

    corner_x, corner_y = (1, 1)
    if random:
        corner_x = np.random.randint(0, 2)
        corner_y = np.random.randint(0, 2)

    dist = [(new_pattern.shape[0] - 2 * gap) * corner_x, (new_pattern.shape[1] - 2 * gap) * corner_y]
    new_pattern[dist[0]:dist[0] + 2 * gap, dist[1]: dist[1] + 2 * gap] = 0
    
    # place a hole in each of the walls

    h1 = (gap + (3 * gap + pattern.shape[0])) // 2
    h2 = ((4 * gap + pattern.shape[0]) + (new_pattern.shape[0] - 2 * gap)) // 2
    h3 = (gap + (3 * gap + pattern.shape[1])) // 2
    h4 = ((4 * gap + pattern.shape[1]) + (new_pattern.shape[1] - 2 * gap)) // 2
    
    if random:
        h1 = np.random.randint(gap, 3 * gap + pattern.shape[0])
        h2 = np.random.randint(4 * gap + pattern.shape[0], new_pattern.shape[0] - 2 * gap)
        h3 = np.random.randint(gap, 3 * gap + pattern.shape[1])
        h4 = np.random.randint(4 * gap + pattern.shape[1], new_pattern.shape[1] - 2 * gap)

    new_pattern[h1:h1 + gap, 3 * gap + pattern.shape[1]:4 * gap + pattern.shape[1]] = 0
    new_pattern[h2:h2 + gap, 3 * gap + pattern.shape[1]:4 * gap + pattern.shape[1]] = 0
    new_pattern[3 * gap + pattern.shape[0]:4 * gap + pattern.shape[0], h3:h3 + gap] = 0
    new_pattern[3 * gap + pattern.shape[0]:4 * gap + pattern.shape[0], h4:h4 + gap] = 0
    
    plt.imshow(new_pattern)
    
    return new_pattern

def create_pyramid(pattern, depth=3, random=False):
    if depth == 0:
        return np.stack([pattern, pattern, pattern, pattern])
    
    patterns = []
    gap = depth
    
    for i in range(4):
        patterns.append(create_pattern(create_pyramid(pattern, depth - 1), gap, random))
    
    return np.stack(patterns)

gap = 1
pattern = np.array([
    [0, 0],
    [0, 0]
])

maze = create_pyramid(pattern, 3)