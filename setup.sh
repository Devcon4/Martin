# File used to setup needed dev packages and tools for the project

# Reset
Color_Off='\033[0m'       # Text Reset

# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

pkgs='libc6 libstdc++6 libvulkan1 build-essential nvidia-driver-550-open vulkan-tools vulkan-sdk libvulkan-dev vulkan-validationlayers-dev spirv-tools libglfw3-dev libglm-dev libxxf86vm-dev libxi-dev gdb cmake'

wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
sudo apt update

echo "${Cyan}Installing packages...${Color_Off}"
for REQUIRED_PKG in $pkgs; do
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "${Yellow}No $REQUIRED_PKG. Setting up $REQUIRED_PKG.${Color_Off}"
  sudo apt --yes install $REQUIRED_PKG
fi
done

[ ! -f "/usr/local/bin/glslc" ] && {
  echo "${Yellow}Installing glslc...${Color_Off}"
  rm -rf /tmp/glslc;
  mkdir -p /tmp/glslc/install/;
  wget -P /tmp/glslc/ "https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/linux/continuous_gcc_release/396/20221222-051244/install.tgz";
  tar -xvzf /tmp/glslc/install.tgz -C /tmp/glslc/;
  sudo cp /tmp/glslc/install/bin/glslc /usr/local/bin/glslc;
}

[ ! -f "./external/stb_image.h" ] && {
  echo "${Yellow}Installing stb_image.h...${Color_Off}";
  wget -P ./external/ "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h";
}
[ ! -f "./external/stb_image_write.h" ] && {
  echo "${Yellow}Installing stb_image_write.h...${Color_Off}";
  wget -P ./external/ "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h";
}
[ ! -f "./external/json.hpp" ] && {
  echo "${Yellow}Installing json.hpp...${Color_Off}";
  wget -P ./external/ "https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp";
}
[ ! -f "./external/tiny_gltf.h" ] && {
  echo "${Yellow}Installing tiny_gltf...${Color_Off}";
  wget -P ./external/ "https://raw.githubusercontent.com/syoyo/tinygltf/release/tiny_gltf.h";
}
[ ! -f "./external/vk_mem_alloc.h" ] && {
  echo "${Yellow}Installing vulkan memory allocator...${Color_Off}";
  wget -P ./external/ "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/master/include/vk_mem_alloc.h";
}

[ ! -f "./external/uuid.h" ] && {
  echo "${Yellow}Installing uuid...${Color_Off}";
  wget -P ./external/ "https://github.com/mariusbancila/stduuid/releases/download/v1.2.3/uuid.h";
}

echo "${Cyan}Done!${Color_Off}"