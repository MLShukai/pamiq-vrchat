# PAMIQ VRChat

Interface for PAMIQ to interact with VRChat on Linux.

## Installation

First, install [**inputtino**](https://github.com/games-on-whales/inputtino/tree/stable/bindings/python#installation) which is a required dependency.

```sh
# Install via pip
pip install pamiq-vrchat

# Install from source
git clone https://github.com/MLShukai/pamiq-vrchat.git
cd pamiq-vrchat
pip install .
```

## Setup VRChat Environment

### Prerequisites

- Linux with Desktop Environment
- Machine capable of running VRChat

### Install Steam

Download and install Steam from the [official website](https://store.steampowered.com/about/).

### Enable Proton

Open Steam → Settings → Compatibility and enable `Enable Steam Play for all other titles`.

![steam_compatibility](./docs/images/steam_compatibility.png)

#### (Optional) Install Proton GE

If you want to use video players in VRChat on Linux, install [Proton GE](https://github.com/GloriousEggroll/proton-ge-custom?tab=readme-ov-file#installation).

After installation, select `GE-Proton` as the compatibility tool in Steam → Settings → Compatibility under `Run other titles with:`.

### Install VRChat

Add **VRChat** to your library from the [Steam store](https://store.steampowered.com/app/438100/VRChat/) and install it.

### Setup OBS

For OBS installation and virtual camera setup, refer to [pamiq-io documentation](https://github.com/MLShukai/pamiq-io?tab=readme-ov-file#obs-virtual-camera).

> \[!IMPORTANT\]
> Don't forget to install `v4l-utils`

> \[!NOTE\]
> The `Output (Scaled) Resolution` and `FPS Value` in OBS Video settings will affect the output of the `ImageSensor` class.
> ![obs-video-setting](./docs/images/obs_video_setting.png)

Capture the VRChat window in OBS and enable the virtual camera.

You can also use our pre-configured [Scene Collection](./obs_settings/VRChatSceneCollection.json). Import it from the OBS `Scene Collection` tab → `Import`, and ensure the checkbox is checked.
