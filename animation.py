import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

raw_adjacency_list = {
    "lAnkle": ["lHeel", "lBigToe", "lKnee"],
    # "lBigToe": ["lAnkle", "lSmallToe"],
    # "lEar": ["lEye", "lShoulder"],
    "lElbow": ["lShoulder", "lWrist"],
    # "lEye": ["lEar", "nose"],
    # "lHeel": ["lAnkle"],
    "lHip": ["lKnee", "midHip"],
    "lKnee": ["lHip", "lAnkle"],
    # "lPinky": ["lWrist"],
    "lShoulder": ["lElbow", "neck", "lEar"],
    # "lSmallToe": ["lBigToe"],
    # "lThumb": ["lWrist"],
    "lWrist": ["lElbow", "lPinky", "lThumb"],
    "midHip": ["rHip", "lHip", "neck"],
    "neck": ["nose", "lShoulder", "rShoulder", "midHip"],
    "nose": ["neck", "lEye", "rEye"],
    "rAnkle": ["rHeel", "rBigToe", "rKnee"],
    # "rBigToe": ["rAnkle", "rSmallToe"],
    # "rEar": ["rEye", "rShoulder"],
    "rElbow": ["rShoulder", "rWrist"],
    # "rEye": ["rEar", "nose"],
    # "rHeel": ["rAnkle"],
    "rHip": ["rKnee", "midHip"],
    "rKnee": ["rHip", "rAnkle"],
    # "rPinky": ["rWrist"],
    "rShoulder": ["rElbow", "neck", "rEar"],
    # "rSmallToe": ["rBigToe"],
    # "rThumb": ["rWrist"],
    "rWrist": ["rElbow", "rPinky", "rThumb"],
}

LABELS = [
    "lAnkle",
    # "lBigToe",
    # "lEar",
    "lElbow",
    # "lEye",
    # "lHeel",
    "lHip",
    "lKnee",
    # "lPinky",
    "lShoulder",
    # "lSmallToe",
    # "lThumb",
    "lWrist",
    "midHip",
    "neck",
    "nose",
    "rAnkle",
    # "rBigToe",
    # "rEar",
    "rElbow",
    # "rEye",
    # "rHeel",
    "rHip",
    "rKnee",
    # "rPinky",
    "rShoulder",
    # "rSmallToe",
    # "rThumb",
    "rWrist"
]

def translate_to_indices(_adjacency_list, labels=LABELS):
    label_to_index = {label: i for i, label in enumerate(labels)}

    translated_adjacency_list = {}

    for key, value in _adjacency_list.items():
        translated_key = label_to_index[key]
        translated_value = [label_to_index[label] for label in value if label in label_to_index]
        translated_adjacency_list[translated_key] = translated_value

    return translated_adjacency_list

adjacency_list = translate_to_indices(raw_adjacency_list)

def animate(*sequences):
    fig = plt.figure(figsize=(20, 10))

    artists = []
    for i, sequence in enumerate(sequences):
        # Create 3D axis
        ax = fig.add_subplot(1, len(sequence), i+1, projection='3d')

        # Scatter plot initialization
        points = ax.scatter([], [], [], c='blue', marker='o')

        # Line plot initialization
        lines = {
            start_point: [ax.plot([], [], [], color='black')[0] for _ in range(len(end_points))]
                for start_point, end_points in adjacency_list.items()
        }

        # Set the limits of the plot
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        artists.append((points, lines))

    def update(frame):
        for sequence, (points, lines) in zip(sequences, artists):
            points._offsets3d = sequence[frame, :, 0], sequence[frame, :, 1], sequence[frame, :, 2]

            for start_point, end_points in adjacency_list.items():
                for i, end_point in enumerate(end_points):
                    x = [sequence[frame, start_point, 0], sequence[frame, end_point, 0]]
                    y = [sequence[frame, start_point, 1], sequence[frame, end_point, 1]]
                    z = [sequence[frame, start_point, 2], sequence[frame, end_point, 2]]
    
                    lines[start_point][i].set_data(x, y)
                    lines[start_point][i].set_3d_properties(z)

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(sequences[0]), interval=100, blit=False)

    plt.close(fig)
    return animation