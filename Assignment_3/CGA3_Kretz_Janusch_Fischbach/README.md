# Assignment 3 - Visual Computing

## Setup

Create an build dir and initialize `cmake`:

```bash
mkdir build && cd build && cmake ..
```

To build the binary use the by `cmake` generated `Makefile` in the build dir:

```bash
make
```

To run the binary use

```bash
cd bin && ./assignment_03
```

> **Important:** It is important that the binary is run from the `build/bin` dir.

## Usage

Run the program with the setup described above. The program should open a glfw
window and display the ornithopter "parked" on the sand in front of the camera.
After startup the ornithopter should not be possible to control. To test this
use `W/A/S/D (E/Q; SHIFT/SPACE)`; the ornithopter should stay parked. When the
program is started the camera is in an orbital configuration. There is 3 camera
implemenations available, with the functionality and reachability explained in
the table below. Additional keybindings provided are:

| Key | Functionality |
| --- | ------------- |
| `ESC`| Closes the window and destroys the allocations properly. |
| `P` | Takes a screenshot and places the image in `build/bin` |
| `F`| Starts the ornithopter "engine", starting the animation and unlocks the ornithopter controls when it has taken off. |
| `1` | Normal Camera mode (default). |
| `2` | Follow camera mode. In this mode the camera is an orbital camera, looking at the ornithopter. When not zoomed or panned, the camera will follow the ornithopter. |
| `3` | Chase camera mode. In this mode the camera is locked behind the tail of the ornithopter, following it. |

> **Note:** In the Normal and Follow camera mode, panning and zooming is enabled. Try it out.

### Movement Keybinds

| Key | Functionality |
| --- | ------------- |
| `W` | Move ornithopter forward. |
| `S` | Move ornithopter backward. |
| `A` | Move ornithopter left. |
| `D` | Move ornithopter right. |
| `Q` | Rotate ornithopter left around the vertical axis. |
| `E` | Rotate ornithopter right around the vertical axis. |
| `Space` | Fly ornithopter up. |
| `SHIFT` | Fly ornithopter down. |

> **Note:** Unlocked when the ornithopters engines are started.

> **Note:** `W/S` movement, pitches the ornithopter. `A/D` movement, rolls the
ornithoper. The maximum tilt (pitch or roll) is 10°.
