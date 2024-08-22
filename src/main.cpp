
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <spdlog/spdlog.h>
#include <stdio.h>

// #include "fmod_errors.h"
// #include <uuid/uuid.h>
#include <vulkan/vulkan_core.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

// #include "fmod.hpp"

constexpr int MAX_FRAMES = 2;

struct MartinImage {
  VkImage image;
  VkDeviceMemory memory;
  VkImageView view;
  VkImageLayout layout;
  VkSampler sampler;
};

struct Init {
  GLFWwindow* window;
  vkb::Instance instance;
  vkb::InstanceDispatchTable inst_disp;
  VkSurfaceKHR surface;
  vkb::Device device;
  vkb::DispatchTable disp;
  vkb::Swapchain swapchain;
};

struct RenderData {
  VkQueue graphics_queue;
  VkQueue present_queue;

  std::vector<VkImage> swapchain_images;
  std::vector<VkImageView> swapchain_image_views;
  std::vector<MartinImage> depth_images;
  std::vector<VkFramebuffer> framebuffers;
  std::array<VkRenderingAttachmentInfoKHR, MAX_FRAMES> color_infos;
  std::array<VkRenderingAttachmentInfoKHR, MAX_FRAMES> depth_infos;
  std::array<VkRenderingInfoKHR, MAX_FRAMES> rendering_infos;

  VkRenderPass render_pass;
  VkPipelineLayout pipeline_layout;
  VkPipeline graphics_pipeline;

  VkCommandPool command_pool;
  std::vector<VkCommandBuffer> command_buffers;

  std::vector<VkSemaphore> available_semaphores;
  std::vector<VkSemaphore> finished_semaphore;
  std::vector<VkFence> in_flight_fences;
  std::vector<VkFence> images_in_flight;

  size_t current_frame = 0;
};

// struct SoundData {
//   FMOD::System* system;
//   std::vector<FMOD::Sound*> sounds;
//   FMOD::Channel* channel = nullptr;
//   FMOD::DSP* dsp;
// };

// uuids::uuid generate_uuid() {
//   std::random_device rd;
//   auto seed_data = std::array<int, std::mt19937::state_size>{};
//   std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
//   std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
//   std::mt19937 generator(seq);
//   uuids::uuid_random_generator gen{generator};

//   return gen();
// }

// bool fmod_error(FMOD_RESULT result) {
//   if (result != FMOD_OK) {
//     spdlog::error("FMOD error: {}", FMOD_ErrorString(result));
//   }

//   return result != FMOD_OK;
// }

// int setup_fmod(SoundData& data) {
//   data.sounds.reserve(48);

//   FMOD::System* system;
//   if (fmod_error(FMOD::System_Create(&system))) {
//     spdlog::error("Failed to create FMOD system");
//     return -1;
//   }

//   if (fmod_error(system->init(512, FMOD_INIT_NORMAL, nullptr))) {
//     spdlog::error("Failed to initialize FMOD system");
//     return -1;
//   }

//   data.system = system;

//   return 0;
// }

// void cleanup_fmod(SoundData& data) {
//   for (auto sound : data.sounds) {
//     if (sound) {
//       sound->release();
//     }
//   }

//   if (data.channel) {
//     data.channel->stop();
//   }

//   if (data.system) {
//     data.system->release();
//   }
// }

// int load_sound(SoundData& data, const std::string& filename) {
//   FMOD::Sound* sound;
//   if (fmod_error(data.system->createSound(filename.c_str(), FMOD_DEFAULT,
//                                           nullptr, &sound))) {
//     spdlog::error("Failed to load sound: {}", filename);
//     return -1;
//   }

//   data.sounds.push_back(sound);

//   spdlog::error("Failed to load sound: {}", filename);
//   return -1;
// }

GLFWwindow* create_window_glfw(const char* window_name = "",
                               bool resize = true) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  if (!resize) glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  GLFWwindow* window =
      glfwCreateWindow(800, 600, window_name, nullptr, nullptr);

  // Set default key callbacks
  glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode,
                                int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
  });

  return window;
}

void destory_window_glfw(GLFWwindow* window) {
  glfwDestroyWindow(window);
  glfwTerminate();
}

VkSurfaceKHR create_surface_glfw(VkInstance instance, GLFWwindow* window,
                                 VkAllocationCallbacks* allocator = nullptr) {
  VkSurfaceKHR surface = VK_NULL_HANDLE;

  VkResult err = glfwCreateWindowSurface(instance, window, allocator, &surface);
  if (err) {
    const char* error_msg;
    int ret = glfwGetError(&error_msg);
    if (ret != 0) {
      spdlog::error("GLFW Error {}: {}", ret,
                    error_msg ? error_msg : "unknown error");
    }
    surface = VK_NULL_HANDLE;
  }
  return surface;
}

std::vector<const char*> instanceExtensions{
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
};

std::vector<const char*> deviceExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
};

int device_initialization(Init& init) {
  init.window = create_window_glfw("Martin Visualizer", true);

  vkb::InstanceBuilder instance_builder;
  auto instance_ret = instance_builder.use_default_debug_messenger()
                          .request_validation_layers()
                          .require_api_version(1, 3, 0)
                          .set_app_name("Martin Visualizer")
                          .set_engine_name("Martin Engine")
                          .enable_extensions(instanceExtensions)
                          .build();
  if (!instance_ret) {
    spdlog::error("Failed to create Vulkan instance: {}",
                  instance_ret.error().message());
    return -1;
  }

  init.instance = instance_ret.value();
  init.inst_disp = init.instance.make_table();

  init.surface = create_surface_glfw(init.instance, init.window);

  vkb::PhysicalDeviceSelector phys_device_selector(init.instance);
  auto phys_device_ret =
      phys_device_selector.add_required_extensions(deviceExtensions)
          .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
          .set_surface(init.surface)
          .select();

  if (!phys_device_ret) {
    spdlog::error("Failed to select Vulkan physical device: {}",
                  phys_device_ret.error().message());
    return -1;
  }

  vkb::PhysicalDevice physical_device = phys_device_ret.value();

  vkb::DeviceBuilder device_builder{physical_device};

  VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = VK_TRUE,
  };

  auto device_ret =
      device_builder.add_pNext(&dynamic_rendering_features).build();
  if (!device_ret) {
    spdlog::error("Failed to create Vulkan device: {}",
                  device_ret.error().message());
    return -1;
  }

  init.device = device_ret.value();
  init.disp = init.device.make_table();

  return 0;
}

struct MartinImageCreateInfo {
  uint32_t width;
  uint32_t height;
  VkFormat format;
  VkImageUsageFlags usage;
  VkMemoryPropertyFlags properties;
  VkImageTiling tiling;
  VkMemoryPropertyFlags memory_properties;
};

uint32_t find_memory_type(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

MartinImage create_image(Init& init, MartinImageCreateInfo& info) {
  MartinImage image = {
      .image = VK_NULL_HANDLE,
      .memory = VK_NULL_HANDLE,
      .view = VK_NULL_HANDLE,
      .layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .sampler = VK_NULL_HANDLE,
  };

  VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = info.format,
      .extent = {.width = info.width, .height = info.height, .depth = 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = info.tiling,
      .usage = info.usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  if (init.disp.createImage(&image_info, nullptr, &image.image) != VK_SUCCESS) {
    spdlog::error("Failed to create image");
    return image;
  }

  VkMemoryRequirements mem_requirements;
  init.disp.getImageMemoryRequirements(image.image, &mem_requirements);

  VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mem_requirements.size,
      .memoryTypeIndex = find_memory_type(init.device.physical_device,
                                          mem_requirements.memoryTypeBits,
                                          info.memory_properties),
  };

  if (init.disp.allocateMemory(&alloc_info, nullptr, &image.memory) !=
      VK_SUCCESS) {
    spdlog::error("Failed to allocate image memory");
    return image;
  }

  init.disp.bindImageMemory(image.image, image.memory, 0);

  return image;
}

int create_image_view(Init& init, MartinImage& image, VkFormat format,
                      VkImageAspectFlags aspect_flags) {
  VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {.aspectMask = aspect_flags,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
  };

  if (init.disp.createImageView(&view_info, nullptr, &image.view) !=
      VK_SUCCESS) {
    spdlog::error("Failed to create image view");
    return -1;
  }

  return 0;
}

int create_image_sampler(Init& init, MartinImage& image) {
  VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_TRUE,
      .maxAnisotropy = 16,
      .compareEnable = VK_FALSE,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .minLod = 0.0f,
      .maxLod = 0.0f,
      .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE,
  };

  if (init.disp.createSampler(&sampler_info, nullptr, &image.sampler) !=
      VK_SUCCESS) {
    spdlog::error("Failed to create image sampler");
    return -1;
  }

  return 0;
}

void destroy_image(Init& init, MartinImage& image) {
  if (image.sampler != VK_NULL_HANDLE) {
    init.disp.destroySampler(image.sampler, nullptr);
  }

  if (image.view != VK_NULL_HANDLE) {
    init.disp.destroyImageView(image.view, nullptr);
  }

  if (image.image != VK_NULL_HANDLE) {
    init.disp.destroyImage(image.image, nullptr);
  }

  if (image.memory != VK_NULL_HANDLE) {
    init.disp.freeMemory(image.memory, nullptr);
  }
}

int create_depth_images(Init& init, RenderData& data) {
  data.depth_images.reserve(MAX_FRAMES);
  data.depth_images.clear();

  auto depth_image_info = MartinImageCreateInfo{
      .width = init.swapchain.extent.width,
      .height = init.swapchain.extent.height,
      .format = VK_FORMAT_D32_SFLOAT,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    MartinImage depth_image = create_image(init, depth_image_info);
    if (depth_image.image == VK_NULL_HANDLE) {
      spdlog::error("Failed to create depth image");
      return -1;
    }

    data.depth_images.push_back(std::move(depth_image));
  }

  return 0;
}

int create_swapchain(Init& init) {
  vkb::SwapchainBuilder swapchain_builder{init.device};
  auto swap_ret = swapchain_builder.set_old_swapchain(init.swapchain)
                      .use_default_image_usage_flags()
                      .build();

  if (!swap_ret) {
    spdlog::error("Failed to create Vulkan swapchain: {}",
                  swap_ret.error().message());
    return -1;
  }

  vkb::destroy_swapchain(init.swapchain);
  init.swapchain = swap_ret.value();
  return 0;
}

int get_queues(Init& init, RenderData& data) {
  auto gq = init.device.get_queue(vkb::QueueType::graphics);
  if (!gq.has_value()) {
    spdlog::error("Failed to get graphics queue: {}", gq.error().message());
    return -1;
  }

  data.graphics_queue = gq.value();

  auto pq = init.device.get_queue(vkb::QueueType::present);
  if (!pq.has_value()) {
    spdlog::error("Failed to get present queue: {}", pq.error().message());
    return -1;
  }

  data.present_queue = pq.value();
  return 0;
}

// int create_render_pass(Init& init, RenderData& data) {
//   VkAttachmentDescription color_attachment = {
//       .format = init.swapchain.image_format,
//       .samples = VK_SAMPLE_COUNT_1_BIT,
//       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
//       .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
//       .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
//       .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
//       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
//       .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
//   };

//   VkAttachmentReference color_attachment_ref = {
//       .attachment = 0,
//       .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
//   };

//   VkSubpassDescription subpass = {
//       .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
//       .colorAttachmentCount = 1,
//       .pColorAttachments = &color_attachment_ref,
//   };

//   VkSubpassDependency dependency = {
//       .srcSubpass = VK_SUBPASS_EXTERNAL,
//       .dstSubpass = 0,
//       .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
//       .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
//       .srcAccessMask = 0,
//       .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
//                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
//   };

//   VkRenderPassCreateInfo render_pass_info = {
//       .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
//       .attachmentCount = 1,
//       .pAttachments = &color_attachment,
//       .subpassCount = 1,
//       .pSubpasses = &subpass,
//       .dependencyCount = 1,
//       .pDependencies = &dependency,
//   };

//   if (init.disp.createRenderPass(&render_pass_info, nullptr,
//                                  &data.render_pass) != VK_SUCCESS) {
//     spdlog::error("Failed to create render pass");
//     return -1;
//   }

//   return 0;
// }

struct MartinImageTransition {
  VkImageLayout old_layout;
  VkImageLayout new_layout;
  VkAccessFlags src_access;
  VkAccessFlags dst_access;
  VkPipelineStageFlags src_stage;
  VkPipelineStageFlags dst_stage;
};

const std::vector<MartinImageTransition> transitions = {
    // Undefined -> Transfer Destination
    {VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
     VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
     VK_PIPELINE_STAGE_TRANSFER_BIT},
    // Transfer Destination -> Shader Read Only
    {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
     VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
    // Shader Read Only -> Transfer Destination
    {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
     VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
     VK_PIPELINE_STAGE_TRANSFER_BIT},
    // Transfer Destination -> Present
    {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT},
    // Transfer Destination -> Depth Attachment
    {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
     VK_ACCESS_TRANSFER_WRITE_BIT,
     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_TRANSFER_BIT,
     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT},
    // Depth Attachment -> Shader Read Only
    {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
     VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
    // Transfer Destination -> Color Attachment
    {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
     VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_TRANSFER_BIT,
     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
    // Transfer Destination -> Attachment
    {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
     VK_ACCESS_TRANSFER_WRITE_BIT,
     VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_TRANSFER_BIT,
     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
    // Color Attachment -> Present
    {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
     VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
     VK_ACCESS_MEMORY_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT},
    // Depth Attachment -> Present
    {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
     VK_ACCESS_MEMORY_READ_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT},
    // undefined -> Depth Attachment
    {VK_IMAGE_LAYOUT_UNDEFINED,
     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0,
     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT},
    // undefined -> Color Attachment
    {VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
     VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
};

// Transition image barrier helper function
void transition_image_layout(
    Init& init, VkCommandBuffer command_buffer, VkImage image,
    VkImageLayout old_layout, VkImageLayout new_layout,
    VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT) {
  VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {.aspectMask = aspect_mask,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
  };

  VkPipelineStageFlags source_stage;
  VkPipelineStageFlags destination_stage;
  bool found = false;

  for (const auto& transition : transitions) {
    if (transition.old_layout == old_layout &&
        transition.new_layout == new_layout) {
      barrier.srcAccessMask = transition.src_access;
      barrier.dstAccessMask = transition.dst_access;
      source_stage = transition.src_stage;
      destination_stage = transition.dst_stage;
      found = true;
      break;
    }
  }

  if (!found) {
    spdlog::error("Failed to find image transition");
    return;
  }

  init.disp.cmdPipelineBarrier(command_buffer, source_stage, destination_stage,
                               0, 0, nullptr, 0, nullptr, 1, &barrier);
}

std::vector<char> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    spdlog::error("Failed to open file: {}", filename);
    throw std::runtime_error("Failed to open file: " + filename);
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

  file.close();

  return buffer;
}

VkShaderModule createShaderModule(Init& init, const std::vector<char>& code) {
  VkShaderModuleCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const uint32_t*>(code.data()),
  };

  VkShaderModule shaderModule;

  if (init.disp.createShaderModule(&createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    spdlog::error("Failed to create shader module");
    throw std::runtime_error("Failed to create shader module");
  }

  return shaderModule;
}

int create_pipeline(Init& init, RenderData& data) {
  auto vert_code = readFile("./resources/shaders/triangle.vert.spv");
  auto frag_code = readFile("./resources/shaders/triangle.frag.spv");

  VkShaderModule vert_module = createShaderModule(init, vert_code);
  VkShaderModule frag_module = createShaderModule(init, frag_code);

  if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
    spdlog::error("Failed to create shader modules");
    return -1;
  }

  VkPipelineShaderStageCreateInfo vert_shader_stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vert_module,
      .pName = "main",
  };

  VkPipelineShaderStageCreateInfo frag_shader_stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = frag_module,
      .pName = "main",
  };

  VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info,
                                                     frag_shader_stage_info};

  VkPipelineVertexInputStateCreateInfo vertex_input_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 0,
      .vertexAttributeDescriptionCount = 0,
  };

  VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
  };

  VkViewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)init.swapchain.extent.width,
      .height = (float)init.swapchain.extent.height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };

  VkRect2D scissor = {
      .offset = {0, 0},
      .extent = init.swapchain.extent,
  };

  VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
  };

  VkPipelineRasterizationStateCreateInfo rasterizer = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .lineWidth = 1.0f,
  };

  VkPipelineMultisampleStateCreateInfo multisampling = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
  };

  VkPipelineColorBlendAttachmentState color_blend_attachment = {
      .blendEnable = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };

  VkPipelineColorBlendStateCreateInfo color_blending = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = &color_blend_attachment,
      .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
  };

  // TODO: pull this out.
  VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 0,
      .pushConstantRangeCount = 0,
  };

  if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr,
                                     &data.pipeline_layout) != VK_SUCCESS) {
    spdlog::error("Failed to create pipeline layout");
    return -1;
  }

  VkPipelineRenderingCreateInfo pipeline_rendering_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &init.swapchain.image_format,
      // .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
  };

  VkGraphicsPipelineCreateInfo pipeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &pipeline_rendering_info,
      .stageCount = 2,
      .pStages = shader_stages,
      .pVertexInputState = &vertex_input_info,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = nullptr,
      .pColorBlendState = &color_blending,
      .layout = data.pipeline_layout,
      .renderPass = VK_NULL_HANDLE,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE,
  };

  if (init.disp.createGraphicsPipelines(
          VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr,
          &data.graphics_pipeline) != VK_SUCCESS) {
    spdlog::error("Failed to create graphics pipeline");
    return -1;
  }

  return 0;
}

// int create_graphics_pipeline(Init& init, RenderData& data) {
//   auto vert_code = readFile("./resources/shaders/triangle.vert.spv");
//   auto frag_code = readFile("./resources/shaders/triangle.frag.spv");

//   VkShaderModule vert_module = createShaderModule(init, vert_code);
//   VkShaderModule frag_module = createShaderModule(init, frag_code);

//   if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
//     spdlog::error("Failed to create shader modules");
//     return -1;
//   }

//   VkPipelineShaderStageCreateInfo vert_shader_stage_info = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
//       .stage = VK_SHADER_STAGE_VERTEX_BIT,
//       .module = vert_module,
//       .pName = "main",
//   };

//   VkPipelineShaderStageCreateInfo frag_shader_stage_info = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
//       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
//       .module = frag_module,
//       .pName = "main",
//   };

//   VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info,
//                                                      frag_shader_stage_info};

//   VkPipelineVertexInputStateCreateInfo vertex_input_info = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
//       .vertexBindingDescriptionCount = 0,
//       .vertexAttributeDescriptionCount = 0,
//   };

//   VkPipelineInputAssemblyStateCreateInfo input_assembly = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
//       .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
//       .primitiveRestartEnable = VK_FALSE,
//   };

//   VkViewport viewport = {
//       .x = 0.0f,
//       .y = 0.0f,
//       .width = (float)init.swapchain.extent.width,
//       .height = (float)init.swapchain.extent.height,
//       .minDepth = 0.0f,
//       .maxDepth = 1.0f,
//   };

//   VkRect2D scissor = {
//       .offset = {0, 0},
//       .extent = init.swapchain.extent,
//   };

//   VkPipelineViewportStateCreateInfo viewport_state = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
//       .viewportCount = 1,
//       .pViewports = &viewport,
//       .scissorCount = 1,
//       .pScissors = &scissor,
//   };

//   VkPipelineRasterizationStateCreateInfo rasterizer = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
//       .depthClampEnable = VK_FALSE,
//       .rasterizerDiscardEnable = VK_FALSE,
//       .polygonMode = VK_POLYGON_MODE_FILL,
//       .cullMode = VK_CULL_MODE_BACK_BIT,
//       .frontFace = VK_FRONT_FACE_CLOCKWISE,
//       .depthBiasEnable = VK_FALSE,
//       .lineWidth = 1.0f,
//   };

//   VkPipelineMultisampleStateCreateInfo multisampling = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
//       .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
//       .sampleShadingEnable = VK_FALSE,
//   };

//   VkPipelineColorBlendAttachmentState color_blend_attachment = {
//       .blendEnable = VK_FALSE,
//       .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
//                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
//   };

//   VkPipelineColorBlendStateCreateInfo color_blending = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
//       .logicOpEnable = VK_FALSE,
//       .logicOp = VK_LOGIC_OP_COPY,
//       .attachmentCount = 1,
//       .pAttachments = &color_blend_attachment,
//       .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
//   };

//   VkPipelineLayoutCreateInfo pipeline_layout_info = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
//       .setLayoutCount = 0,
//       .pushConstantRangeCount = 0,
//   };

//   if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr,
//                                      &data.pipeline_layout) != VK_SUCCESS) {
//     spdlog::error("Failed to create pipeline layout");
//     return -1;
//   }

//   std::vector<VkDynamicState> dynamic_states = {
//       VK_DYNAMIC_STATE_VIEWPORT,
//       VK_DYNAMIC_STATE_SCISSOR,
//   };

//   VkPipelineDynamicStateCreateInfo dynamic_state = {
//       .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
//       .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
//       .pDynamicStates = dynamic_states.data(),
//   };

//   VkGraphicsPipelineCreateInfo pipeline_info = {
//       .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
//       .stageCount = 2,
//       .pStages = shader_stages,
//       .pVertexInputState = &vertex_input_info,
//       .pInputAssemblyState = &input_assembly,
//       .pViewportState = &viewport_state,
//       .pRasterizationState = &rasterizer,
//       .pMultisampleState = &multisampling,
//       .pColorBlendState = &color_blending,
//       .pDynamicState = &dynamic_state,
//       .layout = data.pipeline_layout,
//       .renderPass = data.render_pass,
//       .subpass = 0,
//       .basePipelineHandle = VK_NULL_HANDLE,
//   };

//   if (init.disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info,
//                                         nullptr, &data.graphics_pipeline) !=
//       VK_SUCCESS) {
//     spdlog::error("Failed to create graphics pipeline");
//     return -1;
//   }

//   init.disp.destroyShaderModule(vert_module, nullptr);
//   init.disp.destroyShaderModule(frag_module, nullptr);

//   return 0;
// }

// int create_framebuffers(Init& init, RenderData& data) {
//   data.swapchain_images = init.swapchain.get_images().value();
//   data.swapchain_image_views = init.swapchain.get_image_views().value();

//   data.framebuffers.resize(data.swapchain_image_views.size());

//   for (size_t i = 0; i < data.swapchain_image_views.size(); i++) {
//     VkImageView attachments[] = {data.swapchain_image_views[i]};

//     VkFramebufferCreateInfo framebuffer_info = {
//         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
//         .renderPass = data.render_pass,
//         .attachmentCount = 1,
//         .pAttachments = attachments,
//         .width = init.swapchain.extent.width,
//         .height = init.swapchain.extent.height,
//         .layers = 1,
//     };

//     if (init.disp.createFramebuffer(&framebuffer_info, nullptr,
//                                     &data.framebuffers[i]) != VK_SUCCESS) {
//       spdlog::error("Failed to create framebuffer");
//       return -1;
//     }
//   }

//   return 0;
// }

int create_command_pool(Init& init, RenderData& data) {
  VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = 0,
      .queueFamilyIndex =
          init.device.get_queue_index(vkb::QueueType::graphics).value()};

  if (init.disp.createCommandPool(&pool_info, nullptr, &data.command_pool) !=
      VK_SUCCESS) {
    spdlog::error("Failed to create command pool");
    return -1;
  }

  return 0;
}

int create_command_buffers(Init& init, RenderData& data) {
  data.command_buffers.resize(MAX_FRAMES);

  VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = data.command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = (u_int32_t)data.command_buffers.size(),
  };

  if (init.disp.allocateCommandBuffers(
          &alloc_info, data.command_buffers.data()) != VK_SUCCESS) {
    spdlog::error("Failed to allocate command buffers");
    return -1;
  }

  for (size_t i = 0; i < data.command_buffers.size(); i++) {
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    };

    if (init.disp.beginCommandBuffer(data.command_buffers[i], &begin_info) !=
        VK_SUCCESS) {
      spdlog::error("Failed to begin recording command buffer");
      return -1;
    }

    // VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};

    // VkRenderPassBeginInfo render_pass_info = {
    //     .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
    //     .renderPass = data.render_pass,
    //     .framebuffer = data.framebuffers[i],
    //     .renderArea =
    //         {
    //             .offset = {0, 0},
    //             .extent = init.swapchain.extent,
    //         },
    //     .clearValueCount = 1,
    //     .pClearValues = &clear_color,
    // };

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)init.swapchain.extent.width,
        .height = (float)init.swapchain.extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor = {
        .offset = {0, 0},
        .extent = init.swapchain.extent,
    };

    init.disp.cmdSetViewport(data.command_buffers[i], 0, 1, &viewport);
    init.disp.cmdSetScissor(data.command_buffers[i], 0, 1, &scissor);

    transition_image_layout(init, data.command_buffers[i],
                            data.swapchain_images[i], VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    transition_image_layout(init, data.command_buffers[i],
                            data.depth_images[i].image,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_ASPECT_DEPTH_BIT);

    init.disp.cmdBeginRendering(data.command_buffers[i],
                                &data.rendering_infos[i]);

    init.disp.cmdBindPipeline(data.command_buffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              data.graphics_pipeline);

    // init.disp.cmdBindDescriptorSets(data.command_buffers[i],
    //                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
    //                                 data.pipeline_layout, 0, 0, nullptr, 0,
    //                                 nullptr);

    init.disp.cmdDraw(data.command_buffers[i], 3, 1, 0, 0);

    init.disp.cmdEndRendering(data.command_buffers[i]);

    // init.disp.cmdBeginRenderPass(data.command_buffers[i], &render_pass_info,
    //                              VK_SUBPASS_CONTENTS_INLINE);
    // init.disp.cmdBindPipeline(data.command_buffers[i],
    //                           VK_PIPELINE_BIND_POINT_GRAPHICS,
    //                           data.graphics_pipeline);
    // init.disp.cmdDraw(data.command_buffers[i], 3, 1, 0, 0);

    // init.disp.cmdEndRenderPass(data.command_buffers[i]);

    transition_image_layout(init, data.command_buffers[i],
                            data.swapchain_images[i],
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    transition_image_layout(
        init, data.command_buffers[i], data.depth_images[i].image,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_ASPECT_DEPTH_BIT);

    if (init.disp.endCommandBuffer(data.command_buffers[i]) != VK_SUCCESS) {
      spdlog::error("Failed to record command buffer");
      return -1;
    }
  }

  return 0;
}

int create_rendering_info(Init& init, RenderData& data) {
  data.color_infos = std::array<VkRenderingAttachmentInfoKHR, MAX_FRAMES>();
  data.depth_infos = std::array<VkRenderingAttachmentInfoKHR, MAX_FRAMES>();
  data.rendering_infos = std::array<VkRenderingInfoKHR, MAX_FRAMES>();
  data.swapchain_images = init.swapchain.get_images().value();
  data.swapchain_image_views = init.swapchain.get_image_views().value();

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    data.color_infos[i] = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView = data.swapchain_image_views[i],
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = {0.0f, 0.0f, 0.0f, 1.0f}},
    };

    data.depth_infos[i] = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView = data.depth_images[i].view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .clearValue = {.depthStencil = {1.0f, 0}},
    };

    data.rendering_infos[i] = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
        .renderArea = {.offset = {0, 0}, .extent = init.swapchain.extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &data.color_infos[i],
        .pDepthAttachment = &data.depth_infos[i],
    };
  }

  return 0;
}

int create_sync_objects(Init& init, RenderData& data) {
  data.available_semaphores.resize(MAX_FRAMES);
  data.finished_semaphore.resize(MAX_FRAMES);
  data.in_flight_fences.resize(MAX_FRAMES);
  data.images_in_flight.resize(init.swapchain.image_count, VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
  };

  VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    if (init.disp.createSemaphore(&semaphore_info, nullptr,
                                  &data.available_semaphores[i]) !=
            VK_SUCCESS ||
        init.disp.createSemaphore(&semaphore_info, nullptr,
                                  &data.finished_semaphore[i]) != VK_SUCCESS ||
        init.disp.createFence(&fence_info, nullptr,
                              &data.in_flight_fences[i]) != VK_SUCCESS) {
      spdlog::error("Failed to create synchronization objects for a frame");
      return -1;
    }
  }

  return 0;
}

int recreate_swapchain(Init& init, RenderData& data) {
  init.disp.deviceWaitIdle();

  init.disp.destroyCommandPool(data.command_pool, nullptr);

  for (auto framebuffer : data.framebuffers) {
    init.disp.destroyFramebuffer(framebuffer, nullptr);
  }

  init.swapchain.destroy_image_views(data.swapchain_image_views);

  if (0 != create_swapchain(init)) return -1;
  if (0 != create_depth_images(init, data)) return -1;
  // if (0 != create_framebuffers(init, data)) return -1;
  if (0 != create_rendering_info(init, data)) return -1;
  if (0 != create_pipeline(init, data)) return -1;
  // if (0 != create_render_pass(init, data)) return -1;
  // if (0 != create_graphics_pipeline(init, data)) return -1;
  return 0;
}

int draw_frame(Init& init, RenderData& data) {
  init.disp.waitForFences(1, &data.in_flight_fences[data.current_frame],
                          VK_TRUE, UINT64_MAX);
  u_int32_t image_index = 0;
  VkResult result = init.disp.acquireNextImageKHR(
      init.swapchain, UINT64_MAX, data.available_semaphores[data.current_frame],
      VK_NULL_HANDLE, &image_index);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    return recreate_swapchain(init, data);
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    spdlog::error("Failed to acquire swap chain image");
    return -1;
  }

  if (data.images_in_flight[image_index] != VK_NULL_HANDLE) {
    init.disp.waitForFences(1, &data.images_in_flight[image_index], VK_TRUE,
                            UINT64_MAX);
  }

  data.images_in_flight[image_index] =
      data.in_flight_fences[data.current_frame];

  VkSemaphore wait_semaphores[] = {
      data.available_semaphores[data.current_frame]};
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSemaphore signal_semaphores[] = {
      data.finished_semaphore[data.current_frame]};

  VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = wait_semaphores,
      .pWaitDstStageMask = wait_stages,
      .commandBufferCount = 1,
      .pCommandBuffers = &data.command_buffers[image_index],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = signal_semaphores,
  };

  init.disp.resetFences(1, &data.in_flight_fences[data.current_frame]);

  if (init.disp.queueSubmit(data.graphics_queue, 1, &submit_info,
                            data.in_flight_fences[data.current_frame]) !=
      VK_SUCCESS) {
    spdlog::error("Failed to submit draw command buffer");
    return -1;
  }

  VkSwapchainKHR swapchains[] = {init.swapchain};

  VkPresentInfoKHR present_info = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = signal_semaphores,
      .swapchainCount = 1,
      .pSwapchains = swapchains,
      .pImageIndices = &image_index,
  };

  result = init.disp.queuePresentKHR(data.present_queue, &present_info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    return recreate_swapchain(init, data);
  } else if (result != VK_SUCCESS) {
    spdlog::error("Failed to present swap chain image");
    return -1;
  }

  data.current_frame = (data.current_frame + 1) % MAX_FRAMES;
  return 0;
}

void cleanup(Init& init, RenderData& data) {
  init.disp.deviceWaitIdle();

  // cleanup_fmod(sound_data);

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    init.disp.destroySemaphore(data.available_semaphores[i], nullptr);
    init.disp.destroySemaphore(data.finished_semaphore[i], nullptr);
    init.disp.destroyFence(data.in_flight_fences[i], nullptr);
  }

  init.disp.destroyCommandPool(data.command_pool, nullptr);

  for (auto framebuffer : data.framebuffers) {
    init.disp.destroyFramebuffer(framebuffer, nullptr);
  }

  init.disp.destroyPipeline(data.graphics_pipeline, nullptr);
  init.disp.destroyPipelineLayout(data.pipeline_layout, nullptr);
  init.disp.destroyRenderPass(data.render_pass, nullptr);

  init.swapchain.destroy_image_views(data.swapchain_image_views);

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    init.disp.destroyImageView(data.depth_images[i].view, nullptr);
    init.disp.destroyImage(data.depth_images[i].image, nullptr);
    init.disp.freeMemory(data.depth_images[i].memory, nullptr);
  }

  vkb::destroy_swapchain(init.swapchain);
  vkb::destroy_device(init.device);
  vkb::destroy_surface(init.instance, init.surface);
  vkb::destroy_instance(init.instance);

  destory_window_glfw(init.window);
}

// void update_fmod(SoundData& data) { data.system->update(); }

const int TARGET_FPS = 60;

struct TimeData {
  // Stores total elapsed time, delta time, and frames per second
  double total_time = 0.0;
  double delta_time = 0.0;
  double last_time = 0.0;
  double fps = 0.0;
  double fps_delta = 0.0;
  int fps_count = 0;
};

int main() {
  Init init;
  RenderData render_data;
  TimeData time_data;
  // SoundData sound_data;

  if (0 != device_initialization(init)) return -1;
  if (0 != create_swapchain(init)) return -1;
  if (0 != create_depth_images(init, render_data)) return -1;
  if (0 != get_queues(init, render_data)) return -1;
  // if (0 != create_render_pass(init, render_data)) return -1;
  // if (0 != create_graphics_pipeline(init, render_data)) return -1;
  // if (0 != create_framebuffers(init, render_data)) return -1;
  if (0 != create_rendering_info(init, render_data)) return -1;
  if (0 != create_pipeline(init, render_data)) return -1;
  if (0 != create_command_pool(init, render_data)) return -1;
  if (0 != create_command_buffers(init, render_data)) return -1;
  if (0 != create_sync_objects(init, render_data)) return -1;

  // if (0 != setup_fmod(sound_data)) return -1;

  auto start = std::chrono::high_resolution_clock::now();
  auto last_time = start;

  while (!glfwWindowShouldClose(init.window)) {
    time_data.delta_time =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - last_time)
            .count();
    last_time = std::chrono::high_resolution_clock::now();

    time_data.total_time += time_data.delta_time;
    // Only log every second, set fps to average fps over the last second
    time_data.fps_delta += time_data.delta_time;
    time_data.fps_count++;

    if (time_data.fps_delta >= 1000.0) {
      time_data.fps = time_data.fps_count;
      time_data.fps_delta = 0.0;
      time_data.fps_count = 0;

      spdlog::info("T: {} ms \t∆: {} ms \tFPS: {}", time_data.total_time,
                   time_data.delta_time, time_data.fps);
    }

    auto targetTime = 1000.0 / TARGET_FPS;

    glfwPollEvents();
    // update_fmod(sound_data);
    int ret = draw_frame(init, render_data);
    if (ret != 0) {
      spdlog::error("Failed to draw frame");
      break;
    }

    // We need to sleep to keep the frame rate consistent.
    // Use a spin wait loop to keep the frame rate consistent
    while (std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - last_time)
               .count() < targetTime) {
    }
  }

  init.disp.deviceWaitIdle();

  cleanup(init, render_data);
  return 0;
}