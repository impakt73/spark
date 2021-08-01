use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk::{self, SubpassDependency},
};
use imgui::{DrawCmd, DrawCmdParams};

use std::{
    fs::File,
    io::Read,
    path::Path,
    sync::{Arc, Weak},
    time::Duration,
};

use imgui::internal::RawWrapper;

use std::slice;
use vkutil::*;

use crate::render_graph::{RenderGraph, RenderGraphDispatchParams};
use crate::render_util::ConstantDataWriter;

use std::default::Default;
use std::io;
use std::io::Write;

use align_data::include_aligned;
use ultraviolet::int::UVec2;
use ultraviolet::vec::Vec2;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "tools")]
use crate::tools::ShaderCompiler;

use quick_error::quick_error;

use crate::log::*;

/// Selects a physical device from the provided list
fn select_physical_device(physical_devices: &[vk::PhysicalDevice]) -> vk::PhysicalDevice {
    // TODO: Support proper physical device selection
    //       For now, we just use the first device
    physical_devices[0]
}

/// Size of the scratch memory buffer in bytes that is available to each frame
const FRAME_MEMORY_SIZE: u64 = 8 * 1024 * 1024;

/// Number of individual buffer slots available to shaders during a frame
pub(crate) const NUM_BUFFER_SLOTS: u64 = 8;

/// Number of individual texture slots available to shaders during a frame
pub(crate) const NUM_TEXTURE_SLOTS: u64 = NUM_BUFFER_SLOTS;

pub(crate) const NUM_IMAGE_SLOTS: u64 = NUM_BUFFER_SLOTS;

/// Texture slot index associated with the imgui font
const IMGUI_FONT_TEXTURE_SLOT_INDEX: u64 = 0;

/// Texture slot index associated with the render graph output
const RENDER_GRAPH_OUTPUT_TEXTURE_SLOT_INDEX: u64 = 1;

quick_error! {
    #[derive(Debug)]
    enum RendererError {
        ShaderCompilationNotAvailable
    }
}

#[cfg(feature = "tools")]
fn compile_shader(path: &Path) -> Result<Vec<u32>> {
    let dir = path
        .parent()
        .map(|dir_path| dir_path.to_str().unwrap().to_string());
    let mut compiler = ShaderCompiler::new(dir)?;
    compiler.compile_shader(path.to_str().unwrap())
}

#[cfg(not(feature = "tools"))]
fn compile_shader(_path: &Path) -> Result<Vec<u32>> {
    Err(RendererError::ShaderCompilationNotAvailable.into())
}

pub struct CameraInfo {
    pub z_near: f32,
    pub z_far: f32,
    pub fov_degrees: f32,
    pub position: glam::Vec3,
    pub target: glam::Vec3,
}

impl Default for CameraInfo {
    fn default() -> Self {
        Self {
            z_near: 0.1,
            z_far: 1000.0,
            fov_degrees: 40.0,
            position: glam::vec3(30.0, 25.0, 20.0),
            target: glam::vec3(0.0, 5.0, 0.0),
        }
    }
}

struct FrameState {
    #[allow(dead_code)]
    cmd_buffer: vk::CommandBuffer,
    fence: VkFence,
    descriptor_set: vk::DescriptorSet,
    rendering_finished_semaphore: VkSemaphore,
    imgui_vtx_buffer: Option<VkBuffer>,
    imgui_idx_buffer: Option<VkBuffer>,
}

impl FrameState {
    fn new(
        device: &VkDevice,
        _allocator: Weak<vk_mem::Allocator>,
        command_pool: &VkCommandPool,
        descriptor_pool: &VkDescriptorPool,
        descriptor_set_layout: &VkDescriptorSetLayout,
        frame_memory_buffer: &VkBuffer,
        imgui_renderer: &ImguiRenderer,
    ) -> Result<Self> {
        let cmd_buffer = command_pool.allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)?;

        let rendering_finished_semaphore =
            VkSemaphore::new(device.raw(), &vk::SemaphoreCreateInfo::default())?;
        let fence = VkFence::new(
            device.raw(),
            &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
        )?;

        let descriptor_set =
            descriptor_pool.allocate_descriptor_set(descriptor_set_layout.raw())?;

        unsafe {
            device.raw().upgrade().unwrap().update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .buffer_info(&[vk::DescriptorBufferInfo::builder()
                        .buffer(frame_memory_buffer.raw())
                        .offset(0)
                        .range(FRAME_MEMORY_SIZE)
                        .build()])
                    .build()],
                &[],
            );

            let image_infos = (0..NUM_TEXTURE_SLOTS)
                .map(|_| {
                    vk::DescriptorImageInfo::builder()
                        .image_view(imgui_renderer.font_atlas_image_view.raw())
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .build()
                })
                .collect::<Vec<_>>();
            device.raw().upgrade().unwrap().update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_infos)
                    .build()],
                &[],
            );
        }

        Ok(FrameState {
            cmd_buffer,
            fence,
            descriptor_set,
            rendering_finished_semaphore,
            imgui_idx_buffer: None,
            imgui_vtx_buffer: None,
        })
    }
}

pub struct Renderer {
    #[allow(dead_code)]
    imgui_renderer: ImguiRenderer,
    frame_states: Vec<FrameState>,
    constant_writer: ConstantDataWriter,
    frame_memory_buffer: VkBuffer,
    image_available_semaphores: Vec<VkSemaphore>,
    framebuffers: Vec<VkFramebuffer>,
    renderpass: VkRenderPass,
    #[allow(dead_code)]
    cmd_pool: VkCommandPool,
    #[allow(dead_code)]
    sampler: VkSampler,
    pipeline_layout: VkPipelineLayout,
    #[allow(dead_code)]
    descriptor_set_layout: VkDescriptorSetLayout,
    graph_descriptor_set_layout: VkDescriptorSetLayout,
    #[allow(dead_code)]
    descriptor_pool: VkDescriptorPool,
    imgui_pipeline: VkPipeline,
    graph_output_pipeline: VkPipeline,
    #[allow(dead_code)]
    pipeline_cache: VkPipelineCache,
    cur_frame_idx: usize,
    cur_swapchain_idx: usize,
    swapchain_image_views: Vec<VkImageView>,
    swapchain: VkSwapchain,
    allocator: Arc<vk_mem::Allocator>,
    surface: VkSurface,
    device: VkDevice,
    debug_messenger: Option<VkDebugMessenger>,
    instance: VkInstance,
}

impl Renderer {
    pub fn new(
        window: &winit::window::Window,
        enable_validation: bool,
        context: &mut imgui::Context,
    ) -> Result<Self> {
        let instance = VkInstance::new(window, enable_validation)?;

        let debug_messenger = if instance.is_debug_msg_enabled() {
            Some(VkDebugMessenger::new(&instance)?)
        } else {
            None
        };

        let physical_devices = unsafe { instance.raw().enumerate_physical_devices()? };
        let physical_device = select_physical_device(&physical_devices);

        let surface = VkSurface::new(&instance, window)?;

        let device = VkDevice::new(&instance, physical_device, &surface)?;

        let allocator = Arc::new(vk_mem::Allocator::new(&vk_mem::AllocatorCreateInfo {
            physical_device,
            device: (*device.raw().upgrade().unwrap()).clone(),
            instance: instance.raw().clone(),
            flags: vk_mem::AllocatorCreateFlags::NONE,
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        })?);

        let pipeline_cache =
            VkPipelineCache::new(device.raw(), &vk::PipelineCacheCreateInfo::default())?;

        let swapchain = VkSwapchain::new(
            &instance,
            &surface,
            &device,
            window.inner_size().width,
            window.inner_size().height,
            None,
        )?;

        let surface_format = swapchain.surface_format;
        let surface_resolution = swapchain.surface_resolution;
        let desired_image_count = swapchain.images.len() as u32;
        let queue_family_index = 0;

        let swapchain_image_views = swapchain
            .images
            .iter()
            .map(|image| {
                VkImageView::new(
                    device.raw(),
                    &vk::ImageViewCreateInfo::builder()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(
                            vk::ComponentMapping::builder()
                                .r(vk::ComponentSwizzle::IDENTITY)
                                .g(vk::ComponentSwizzle::IDENTITY)
                                .b(vk::ComponentSwizzle::IDENTITY)
                                .a(vk::ComponentSwizzle::IDENTITY)
                                .build(),
                        )
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        ),
                )
            })
            .collect::<Result<Vec<VkImageView>>>()?;

        let renderpass = VkRenderPass::new(
            device.raw(),
            &vk::RenderPassCreateInfo::builder()
                .dependencies(&[SubpassDependency::builder()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .build()])
                .attachments(&[vk::AttachmentDescription::builder()
                    .format(surface_format.format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .build()])
                .subpasses(&[vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&[vk::AttachmentReference::builder()
                        .attachment(0)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .build()])
                    .build()]),
        )?;

        let framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                VkFramebuffer::new(
                    device.raw(),
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(renderpass.raw())
                        .attachments(&[image_view.raw()])
                        .width(surface_resolution.width)
                        .height(surface_resolution.height)
                        .layers(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let cmd_pool = VkCommandPool::new(
            device.raw(),
            &vk::CommandPoolCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
        )?;

        let sampler = VkSampler::new(
            device.raw(),
            &vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .min_lod(0.0)
                .max_lod(10000.0)
                .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK),
        )?;

        let descriptor_set_layout = VkDescriptorSetLayout::new(
            device.raw(),
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::COMPUTE,
                    )
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(&[sampler.raw()])
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(NUM_TEXTURE_SLOTS as u32)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ]),
        )?;

        let graph_descriptor_set_layout = VkDescriptorSetLayout::new(
            device.raw(),
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(NUM_TEXTURE_SLOTS as u32)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(NUM_BUFFER_SLOTS as u32)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ]),
        )?;

        let pipeline_layout = VkPipelineLayout::new(
            device.raw(),
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[
                    descriptor_set_layout.raw(),
                    graph_descriptor_set_layout.raw(),
                ])
                .push_constant_ranges(&[vk::PushConstantRange::builder()
                    .offset(0)
                    .size((4 * std::mem::size_of::<u32>()) as u32)
                    .stage_flags(
                        vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::COMPUTE,
                    )
                    .build()]),
        )?;

        let descriptor_pool = VkDescriptorPool::new(
            device.raw(),
            &vk::DescriptorPoolCreateInfo::builder()
                .max_sets(desired_image_count)
                .pool_sizes(&[
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                        .descriptor_count(desired_image_count)
                        .build(),
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::SAMPLER)
                        .descriptor_count(desired_image_count)
                        .build(),
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(desired_image_count * (NUM_TEXTURE_SLOTS as u32))
                        .build(),
                ]),
        )?;

        let imgui_vert_spv = unsafe {
            include_aligned!(u32, "../spv/ImguiTriangle.vert.spv")
                .align_to::<u32>()
                .1
        };
        let imgui_vert_module = VkShaderModule::new(
            device.raw(),
            &vk::ShaderModuleCreateInfo::builder().code(imgui_vert_spv),
        )?;

        let imgui_frag_spv = unsafe {
            include_aligned!(u32, "../spv/ImguiTriangle.frag.spv")
                .align_to::<u32>()
                .1
        };
        let imgui_frag_module = VkShaderModule::new(
            device.raw(),
            &vk::ShaderModuleCreateInfo::builder().code(imgui_frag_spv),
        )?;

        let imgui_entry_point_c_string = std::ffi::CString::new("main").unwrap();
        let imgui_pipeline = pipeline_cache.create_graphics_pipeline(
            &vk::GraphicsPipelineCreateInfo::builder()
                .stages(&[
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(imgui_vert_module.raw())
                        .name(imgui_entry_point_c_string.as_c_str())
                        .build(),
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(imgui_frag_module.raw())
                        .name(imgui_entry_point_c_string.as_c_str())
                        .build(),
                ])
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                )
                .vertex_input_state(
                    &vk::PipelineVertexInputStateCreateInfo::builder()
                        .vertex_binding_descriptions(&[vk::VertexInputBindingDescription::builder(
                        )
                        .binding(0)
                        .stride(std::mem::size_of::<imgui::DrawVert>() as u32)
                        .input_rate(vk::VertexInputRate::VERTEX)
                        .build()])
                        .vertex_attribute_descriptions(&[
                            vk::VertexInputAttributeDescription::builder()
                                .location(0)
                                .binding(0)
                                .format(vk::Format::R32G32_SFLOAT)
                                .offset(0)
                                .build(),
                            vk::VertexInputAttributeDescription::builder()
                                .location(1)
                                .binding(0)
                                .format(vk::Format::R32G32_SFLOAT)
                                .offset(8)
                                .build(),
                            vk::VertexInputAttributeDescription::builder()
                                .location(2)
                                .binding(0)
                                .format(vk::Format::R32_UINT)
                                .offset(16)
                                .build(),
                        ]),
                )
                .viewport_state(
                    &vk::PipelineViewportStateCreateInfo::builder()
                        .viewports(&[vk::Viewport::default()])
                        .scissors(&[vk::Rect2D::default()]),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::builder()
                        .polygon_mode(vk::PolygonMode::FILL)
                        .cull_mode(vk::CullModeFlags::NONE)
                        .line_width(1.0),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                )
                // Don't need depth state
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
                        vk::PipelineColorBlendAttachmentState::builder()
                            .color_write_mask(
                                vk::ColorComponentFlags::R
                                    | vk::ColorComponentFlags::G
                                    | vk::ColorComponentFlags::B
                                    | vk::ColorComponentFlags::A,
                            )
                            .blend_enable(true)
                            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .color_blend_op(vk::BlendOp::ADD)
                            .src_alpha_blend_factor(vk::BlendFactor::ONE)
                            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                            .alpha_blend_op(vk::BlendOp::ADD)
                            .build(),
                    ]),
                )
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::builder()
                        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
                )
                .layout(pipeline_layout.raw())
                .render_pass(renderpass.raw())
                .subpass(0),
        )?;

        let graph_output_vert_spv = unsafe {
            include_aligned!(u32, "../spv/FullscreenPass.vert.spv")
                .align_to::<u32>()
                .1
        };
        let graph_output_vert_module = VkShaderModule::new(
            device.raw(),
            &vk::ShaderModuleCreateInfo::builder().code(graph_output_vert_spv),
        )?;

        let graph_output_frag_spv = unsafe {
            include_aligned!(u32, "../spv/CopyTexture.frag.spv")
                .align_to::<u32>()
                .1
        };
        let graph_output_frag_module = VkShaderModule::new(
            device.raw(),
            &vk::ShaderModuleCreateInfo::builder().code(graph_output_frag_spv),
        )?;

        let graph_output_entry_point_c_string = std::ffi::CString::new("main").unwrap();
        let graph_output_pipeline = pipeline_cache.create_graphics_pipeline(
            &vk::GraphicsPipelineCreateInfo::builder()
                .stages(&[
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(graph_output_vert_module.raw())
                        .name(graph_output_entry_point_c_string.as_c_str())
                        .build(),
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(graph_output_frag_module.raw())
                        .name(graph_output_entry_point_c_string.as_c_str())
                        .build(),
                ])
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                )
                .vertex_input_state(
                    &vk::PipelineVertexInputStateCreateInfo::builder()
                        .vertex_binding_descriptions(&[])
                        .vertex_attribute_descriptions(&[]),
                )
                .viewport_state(
                    &vk::PipelineViewportStateCreateInfo::builder()
                        .viewports(&[vk::Viewport::default()])
                        .scissors(&[vk::Rect2D::default()]),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::builder()
                        .polygon_mode(vk::PolygonMode::FILL)
                        .cull_mode(vk::CullModeFlags::NONE)
                        .line_width(1.0),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                )
                // Don't need depth state
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
                        vk::PipelineColorBlendAttachmentState::builder()
                            .color_write_mask(
                                vk::ColorComponentFlags::R
                                    | vk::ColorComponentFlags::G
                                    | vk::ColorComponentFlags::B
                                    | vk::ColorComponentFlags::A,
                            )
                            .blend_enable(false)
                            .build(),
                    ]),
                )
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::builder()
                        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
                )
                .layout(pipeline_layout.raw())
                .render_pass(renderpass.raw())
                .subpass(0),
        )?;

        let image_available_semaphores = swapchain
            .images
            .iter()
            .map(|_| VkSemaphore::new(device.raw(), &vk::SemaphoreCreateInfo::default()))
            .collect::<Result<Vec<_>>>()?;

        let frame_memory_buffer = VkBuffer::new(
            Arc::downgrade(&allocator),
            &ash::vk::BufferCreateInfo::builder()
                .size(FRAME_MEMORY_SIZE * (desired_image_count as u64))
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuToGpu,
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            },
        )?;

        let imgui_renderer = ImguiRenderer::new(&device, Arc::downgrade(&allocator), context)?;

        let frame_states = swapchain_image_views
            .iter()
            .map(|_image_view| {
                FrameState::new(
                    &device,
                    Arc::downgrade(&allocator),
                    &cmd_pool,
                    &descriptor_pool,
                    &descriptor_set_layout,
                    &frame_memory_buffer,
                    &imgui_renderer,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Renderer {
            imgui_renderer,
            frame_states,
            constant_writer: ConstantDataWriter::default(),
            frame_memory_buffer,
            framebuffers,
            image_available_semaphores,
            renderpass,
            cmd_pool,
            sampler,
            pipeline_layout,
            descriptor_set_layout,
            graph_descriptor_set_layout,
            descriptor_pool,
            imgui_pipeline,
            graph_output_pipeline,
            pipeline_cache,
            cur_frame_idx: 0,
            cur_swapchain_idx: 0,
            swapchain_image_views,
            swapchain,
            allocator,
            surface,
            device,
            debug_messenger,
            instance,
        })
    }

    fn get_cur_frame_state(&self) -> &FrameState {
        &self.frame_states[self.cur_swapchain_idx]
    }

    pub fn get_cur_swapchain_idx(&self) -> usize {
        self.cur_swapchain_idx
    }

    pub fn get_num_frame_states(&self) -> usize {
        self.frame_states.len()
    }

    pub fn get_swapchain_resolution(&self) -> (u32, u32) {
        (
            self.swapchain.surface_resolution.width,
            self.swapchain.surface_resolution.height,
        )
    }

    pub fn get_swapchain_format(&self) -> vk::Format {
        self.swapchain.surface_format.format
    }

    pub fn recreate_swapchain(&mut self, window: &winit::window::Window) -> Result<()> {
        info!(
            "Recreating {}x{} swapchain!",
            window.inner_size().width,
            window.inner_size().height
        );

        // Make sure all previous rendering work is completed before we destroy the old swapchain resources
        self.wait_for_idle();

        let swapchain = VkSwapchain::new(
            &self.instance,
            &self.surface,
            &self.device,
            window.inner_size().width,
            window.inner_size().height,
            Some(&self.swapchain),
        )?;

        let surface_format = swapchain.surface_format;
        let surface_resolution = swapchain.surface_resolution;

        let swapchain_image_views = swapchain
            .images
            .iter()
            .map(|image| {
                VkImageView::new(
                    self.device.raw(),
                    &vk::ImageViewCreateInfo::builder()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(
                            vk::ComponentMapping::builder()
                                .r(vk::ComponentSwizzle::IDENTITY)
                                .g(vk::ComponentSwizzle::IDENTITY)
                                .b(vk::ComponentSwizzle::IDENTITY)
                                .a(vk::ComponentSwizzle::IDENTITY)
                                .build(),
                        )
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        ),
                )
            })
            .collect::<Result<Vec<VkImageView>>>()?;

        let framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                VkFramebuffer::new(
                    self.device.raw(),
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(self.renderpass.raw())
                        .attachments(&[image_view.raw()])
                        .width(surface_resolution.width)
                        .height(surface_resolution.height)
                        .layers(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        self.swapchain_image_views = swapchain_image_views;
        self.framebuffers = framebuffers;
        self.swapchain = swapchain;

        Ok(())
    }

    pub fn begin_frame(&mut self) {
        unsafe {
            // Acquire the current swapchain image index
            // TODO: Handle suboptimal swapchains
            let (image_index, _is_suboptimal) = self
                .swapchain
                .acquire_next_image(
                    u64::MAX,
                    Some(self.image_available_semaphores[self.cur_frame_idx].raw()),
                    None,
                )
                .unwrap();
            // TODO: This should never happen since we're already handling window resize events, but this could be handled
            // more robustly in the future.
            assert!(!_is_suboptimal);
            self.cur_swapchain_idx = image_index as usize;

            let constant_data_offset = self.cur_swapchain_idx * (FRAME_MEMORY_SIZE as usize);

            self.constant_writer.begin_frame(
                self.frame_memory_buffer
                    .info()
                    .get_mapped_data()
                    .add(constant_data_offset),
                FRAME_MEMORY_SIZE as usize,
            );

            let frame_state = self.get_cur_frame_state();

            let device = self.device.raw().upgrade().unwrap();

            let descriptor_set = frame_state.descriptor_set;

            // Wait for the resources for this frame to become available
            device
                .wait_for_fences(&[frame_state.fence.raw()], true, u64::MAX)
                .unwrap();

            let cmd_buffer = frame_state.cmd_buffer;

            device
                .begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout.raw(),
                0,
                &[descriptor_set],
                &[constant_data_offset as u32],
            );

            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout.raw(),
                0,
                &[descriptor_set],
                &[constant_data_offset as u32],
            );
        }
    }

    pub fn begin_render(&self) {
        let frame_state = self.get_cur_frame_state();
        let framebuffer = &self.framebuffers[self.cur_swapchain_idx];
        unsafe {
            self.device.raw().upgrade().unwrap().cmd_begin_render_pass(
                frame_state.cmd_buffer,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.renderpass.raw())
                    .framebuffer(framebuffer.raw())
                    .render_area(
                        vk::Rect2D::builder()
                            .extent(self.swapchain.surface_resolution)
                            .build(),
                    )
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }]),
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn end_render(&self) {
        let frame_state = self.get_cur_frame_state();
        unsafe {
            self.device
                .raw()
                .upgrade()
                .unwrap()
                .cmd_end_render_pass(frame_state.cmd_buffer);
        }
    }

    pub fn end_frame(&mut self) {
        self.constant_writer.end_frame();

        let frame_state = self.get_cur_frame_state();
        unsafe {
            self.device
                .raw()
                .upgrade()
                .unwrap()
                .end_command_buffer(frame_state.cmd_buffer)
                .unwrap();

            let wait_semaphores = [self.image_available_semaphores[self.cur_frame_idx].raw()];
            let command_buffers = [frame_state.cmd_buffer];
            let signal_semaphores = [frame_state.rendering_finished_semaphore.raw()];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TOP_OF_PIPE])
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();

            let fence = &frame_state.fence;
            self.device
                .raw()
                .upgrade()
                .unwrap()
                .reset_fences(&[fence.raw()])
                .unwrap();
            self.device
                .raw()
                .upgrade()
                .unwrap()
                .queue_submit(self.device.present_queue(), &[submit_info], fence.raw())
                .unwrap();

            let _is_suboptimal = self
                .swapchain
                .present_image(
                    self.cur_swapchain_idx as u32,
                    &signal_semaphores,
                    self.device.present_queue(),
                )
                .unwrap();
            // TODO: This should never happen since we're already handling window resize events, but this could be handled
            // more robustly in the future.
            assert!(!_is_suboptimal);

            self.cur_frame_idx = (self.cur_frame_idx + 1) % self.swapchain.images.len();
        }
    }

    pub fn wait_for_idle(&self) {
        unsafe { self.get_device().device_wait_idle().unwrap() };
    }

    pub fn get_device(&self) -> Arc<ash::Device> {
        self.device.raw().upgrade().unwrap()
    }

    pub fn get_allocator(&self) -> Weak<vk_mem::Allocator> {
        Arc::downgrade(&self.allocator)
    }

    pub fn get_graph_descriptor_set_layout(&self) -> &VkDescriptorSetLayout {
        &self.graph_descriptor_set_layout
    }

    pub fn update_graph_image(&mut self, image_view: &VkImageView) {
        let frame_state = &mut self.frame_states[self.cur_swapchain_idx];

        let device = self.device.raw().upgrade().unwrap();

        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(frame_state.descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(RENDER_GRAPH_OUTPUT_TEXTURE_SLOT_INDEX as u32)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::builder()
                        .image_view(image_view.raw())
                        .image_layout(vk::ImageLayout::GENERAL)
                        .build()])
                    .build()],
                &[],
            );
        }
    }

    pub fn render_graph_image(&mut self) {
        let frame_state = &mut self.frame_states[self.cur_swapchain_idx];

        let device = self.device.raw().upgrade().unwrap();
        let cmd_buffer = frame_state.cmd_buffer;

        let swapchain_resolution = self.swapchain.surface_resolution;

        unsafe {
            device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graph_output_pipeline.raw(),
            );

            device.cmd_set_viewport(
                cmd_buffer,
                0,
                &[vk::Viewport::builder()
                    .x(0.0)
                    .y(0.0)
                    .width(swapchain_resolution.width as f32)
                    .height(swapchain_resolution.height as f32)
                    .build()],
            );

            device.cmd_set_scissor(
                cmd_buffer,
                0,
                &[vk::Rect2D::builder()
                    .offset(vk::Offset2D::builder().x(0).y(0).build())
                    .extent(
                        vk::Extent2D::builder()
                            .width(swapchain_resolution.width)
                            .height(swapchain_resolution.height)
                            .build(),
                    )
                    .build()],
            );

            let push_constant_0 = (RENDER_GRAPH_OUTPUT_TEXTURE_SLOT_INDEX & 0xff) << 24;
            device.cmd_push_constants(
                cmd_buffer,
                self.pipeline_layout.raw(),
                vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::COMPUTE,
                0,
                &push_constant_0.to_le_bytes(),
            );

            device.cmd_draw(cmd_buffer, 6, 1, 0, 0);
        }
    }

    pub fn render_ui(&mut self, draw_data: &imgui::DrawData) {
        let frame_state = &mut self.frame_states[self.cur_swapchain_idx];

        let device = self.device.raw().upgrade().unwrap();
        let cmd_buffer = frame_state.cmd_buffer;

        unsafe {
            let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
            let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
            if (fb_width > 0.0) && (fb_height > 0.0) && draw_data.total_idx_count > 0 {
                let total_vtx_count = draw_data.total_vtx_count as usize;
                let vtx_buffer_size = total_vtx_count * std::mem::size_of::<imgui::DrawVert>();
                let vtx_buffer = VkBuffer::new(
                    Arc::downgrade(&self.allocator),
                    &ash::vk::BufferCreateInfo::builder()
                        .size(vtx_buffer_size as u64)
                        .usage(vk::BufferUsageFlags::VERTEX_BUFFER),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::CpuToGpu,
                        flags: vk_mem::AllocationCreateFlags::MAPPED,
                        ..Default::default()
                    },
                )
                .unwrap();
                let vtx_buffer_slice =
                    slice::from_raw_parts_mut(vtx_buffer.info().get_mapped_data(), vtx_buffer_size);
                let vtx_buffer_raw = vtx_buffer.raw();

                let total_idx_count = draw_data.total_idx_count as usize;
                let idx_buffer_size = total_idx_count * std::mem::size_of::<imgui::DrawIdx>();
                let idx_buffer = VkBuffer::new(
                    Arc::downgrade(&self.allocator),
                    &ash::vk::BufferCreateInfo::builder()
                        .size(idx_buffer_size as u64)
                        .usage(vk::BufferUsageFlags::INDEX_BUFFER),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::CpuToGpu,
                        flags: vk_mem::AllocationCreateFlags::MAPPED,
                        ..Default::default()
                    },
                )
                .unwrap();
                let idx_buffer_slice =
                    slice::from_raw_parts_mut(idx_buffer.info().get_mapped_data(), idx_buffer_size);
                let idx_buffer_raw = idx_buffer.raw();

                let mut vtx_bytes_written: usize = 0;
                let mut vtx_buffer_offsets = Vec::new();

                let mut idx_bytes_written: usize = 0;
                let mut idx_buffer_offsets = Vec::new();

                for draw_list in draw_data.draw_lists() {
                    let vtx_data_src = draw_list.vtx_buffer().as_ptr() as *const u8;
                    let vtx_data_dst =
                        (vtx_buffer_slice.as_mut_ptr() as *mut u8).add(vtx_bytes_written);
                    let vtx_data_size =
                        draw_list.vtx_buffer().len() * std::mem::size_of::<imgui::DrawVert>();
                    core::ptr::copy_nonoverlapping(vtx_data_src, vtx_data_dst, vtx_data_size);
                    vtx_buffer_offsets.push(vtx_bytes_written);
                    vtx_bytes_written += vtx_data_size;

                    let idx_data_src = draw_list.idx_buffer().as_ptr() as *const u8;
                    let idx_data_dst =
                        (idx_buffer_slice.as_mut_ptr() as *mut u8).add(idx_bytes_written);
                    let idx_data_size =
                        draw_list.idx_buffer().len() * std::mem::size_of::<imgui::DrawIdx>();
                    core::ptr::copy_nonoverlapping(idx_data_src, idx_data_dst, idx_data_size);
                    idx_buffer_offsets.push(idx_bytes_written);
                    idx_bytes_written += idx_data_size;
                }

                frame_state.imgui_vtx_buffer = Some(vtx_buffer);
                frame_state.imgui_idx_buffer = Some(idx_buffer);

                device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.imgui_pipeline.raw(),
                );

                let fb_scale = draw_data.framebuffer_scale;
                device.cmd_set_viewport(
                    cmd_buffer,
                    0,
                    &[vk::Viewport::builder()
                        .x(draw_data.display_pos[0] * fb_scale[0])
                        .y(draw_data.display_pos[1] * fb_scale[1])
                        .width(draw_data.display_size[0] * fb_scale[0])
                        .height(draw_data.display_size[1] * fb_scale[1])
                        .build()],
                );

                let clip_off = draw_data.display_pos;
                let clip_scale = draw_data.framebuffer_scale;

                let left = draw_data.display_pos[0];
                let right = draw_data.display_pos[0] + draw_data.display_size[0];
                let top = draw_data.display_pos[1];
                let bottom = draw_data.display_pos[1] + draw_data.display_size[1];
                let matrix = [
                    [(2.0 / (right - left)), 0.0, 0.0, 0.0],
                    [0.0, (2.0 / (top - bottom)), 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [
                        (right + left) / (left - right),
                        (top + bottom) / (bottom - top),
                        0.0,
                        1.0,
                    ],
                ];

                // Identify the current constant buffer offset before we write any new data into it
                let dword_offset = self.constant_writer.dword_offset();

                // Write the imgui matrix into the buffer
                for row in &matrix {
                    for val in row {
                        self.constant_writer.write_all(&val.to_le_bytes()).unwrap();
                    }
                }

                for (idx, draw_list) in draw_data.draw_lists().enumerate() {
                    device.cmd_bind_vertex_buffers(
                        cmd_buffer,
                        0,
                        &[vtx_buffer_raw],
                        &[vtx_buffer_offsets[idx] as u64],
                    );

                    device.cmd_bind_index_buffer(
                        cmd_buffer,
                        idx_buffer_raw,
                        idx_buffer_offsets[idx] as u64,
                        vk::IndexType::UINT16,
                    );

                    for cmd in draw_list.commands() {
                        match cmd {
                            DrawCmd::Elements {
                                count,
                                cmd_params:
                                    DrawCmdParams {
                                        clip_rect,
                                        texture_id,
                                        vtx_offset,
                                        idx_offset,
                                    },
                            } => {
                                let clip_rect = [
                                    (clip_rect[0] - clip_off[0]) * clip_scale[0],
                                    (clip_rect[1] - clip_off[1]) * clip_scale[1],
                                    (clip_rect[2] - clip_off[0]) * clip_scale[0],
                                    (clip_rect[3] - clip_off[1]) * clip_scale[1],
                                ];

                                if clip_rect[0] < fb_width
                                    && clip_rect[1] < fb_height
                                    && clip_rect[2] >= 0.0
                                    && clip_rect[3] >= 0.0
                                {
                                    let scissor_x = f32::max(0.0, clip_rect[0]).floor() as i32;
                                    let scissor_y = f32::max(0.0, clip_rect[1]).floor() as i32;
                                    let scissor_w =
                                        (clip_rect[2] - clip_rect[0]).abs().ceil() as u32;
                                    let scissor_h =
                                        (clip_rect[3] - clip_rect[1]).abs().ceil() as u32;

                                    device.cmd_set_scissor(
                                        cmd_buffer,
                                        0,
                                        &[vk::Rect2D::builder()
                                            .offset(
                                                vk::Offset2D::builder()
                                                    .x(scissor_x)
                                                    .y(scissor_y)
                                                    .build(),
                                            )
                                            .extent(
                                                vk::Extent2D::builder()
                                                    .width(scissor_w)
                                                    .height(scissor_h)
                                                    .build(),
                                            )
                                            .build()],
                                    );

                                    // The texture slot index is stored inside the ImGui texture id
                                    let texture_index: u32 = texture_id.id() as u32;
                                    let push_constant_0 = ((texture_index & 0xff) << 24)
                                        | (dword_offset & 0x00ffffff);
                                    device.cmd_push_constants(
                                        cmd_buffer,
                                        self.pipeline_layout.raw(),
                                        vk::ShaderStageFlags::VERTEX
                                            | vk::ShaderStageFlags::FRAGMENT
                                            | vk::ShaderStageFlags::COMPUTE,
                                        0,
                                        &push_constant_0.to_le_bytes(),
                                    );

                                    device.cmd_draw_indexed(
                                        cmd_buffer,
                                        count as u32,
                                        1,
                                        idx_offset as u32,
                                        vtx_offset as i32,
                                        0,
                                    );
                                }
                            }
                            DrawCmd::ResetRenderState => (), // NOTE: This doesn't seem necessary given how pipelines work?
                            DrawCmd::RawCallback { callback, raw_cmd } => {
                                callback(draw_list.raw(), raw_cmd)
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn create_compute_pipeline_from_buffer(&mut self, spv: &[u32]) -> Result<VkPipeline> {
        let module = VkShaderModule::new(
            self.device.raw(),
            &vk::ShaderModuleCreateInfo::builder().code(spv),
        )?;

        let entry_point_c_string = std::ffi::CString::new("main").unwrap();
        let pipeline = self.pipeline_cache.create_compute_pipeline(
            &vk::ComputePipelineCreateInfo::builder()
                .stage(
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module.raw())
                        .name(entry_point_c_string.as_c_str())
                        .build(),
                )
                .layout(self.pipeline_layout.raw()),
        )?;

        Ok(pipeline)
    }

    pub fn create_compute_pipeline_from_file(&mut self, spv_path: &str) -> Result<VkPipeline> {
        let path = Path::new(spv_path);

        let spv = if let Some("spv") = path.extension().map(|x| x.to_str().unwrap()) {
            let mut buf = Vec::new();
            let mut file = File::open(spv_path)?;
            loop {
                let mut dword: [u8; 4] = [0; 4];

                match file.read(&mut dword)? {
                    4 => {
                        buf.push(u32::from_le_bytes(dword));
                    }
                    0 => {
                        break;
                    }
                    _ => {
                        return Err(Box::new(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "Not enough bytes for SPIR-V data",
                        )));
                    }
                }
            }

            buf
        } else {
            compile_shader(path)?
        };

        self.create_compute_pipeline_from_buffer(&spv)
    }

    pub fn execute_graph(
        &mut self,
        graph: &RenderGraph,
        cur_time: &Duration,
        sync_buffer_data: &[f32],
        camera_info: CameraInfo,
    ) -> Result<()> {
        let renderer_frame_state = &mut self.frame_states[self.cur_swapchain_idx];

        let graph_frame_state = &graph.frame_states[self.cur_swapchain_idx];

        let device = self.device.raw().upgrade().unwrap();
        let cmd_buffer = renderer_frame_state.cmd_buffer;

        let cur_time_bytes = cur_time.as_secs_f32().to_le_bytes();
        self.constant_writer.write_all(&cur_time_bytes).unwrap();

        let swapchain_res_float = Vec2::new(
            self.swapchain.surface_resolution.width as f32,
            self.swapchain.surface_resolution.height as f32,
        );
        self.constant_writer
            .write_all(swapchain_res_float.as_byte_slice())
            .unwrap();

        let swapchain_res_uint = UVec2::new(
            self.swapchain.surface_resolution.width,
            self.swapchain.surface_resolution.height,
        );
        self.constant_writer
            .write_all(swapchain_res_uint.as_byte_slice())
            .unwrap();

        let proj_matrix = glam::Mat4::perspective_rh(
            camera_info.fov_degrees.to_radians(),
            (self.swapchain.surface_resolution.width as f32)
                / (self.swapchain.surface_resolution.height as f32),
            camera_info.z_near,
            camera_info.z_far,
        );

        self.constant_writer
            .write_all(bytemuck::bytes_of(&proj_matrix))
            .unwrap();

        let camera_pos = camera_info.position; //glam::vec3(30.0, 25.0, 20.0);

        let view_matrix =
            glam::Mat4::look_at_rh(camera_pos, camera_info.target, glam::vec3(0.0, 1.0, 0.0));

        self.constant_writer
            .write_all(bytemuck::bytes_of(&view_matrix))
            .unwrap();

        let proj_view_matrix = proj_matrix * view_matrix;

        self.constant_writer
            .write_all(bytemuck::bytes_of(&proj_view_matrix))
            .unwrap();

        // Write sync buffer
        if !sync_buffer_data.is_empty() {
            let sync_buffer_data_as_u8 = unsafe { sync_buffer_data.align_to::<u8>().1 };
            self.constant_writer
                .write_all(sync_buffer_data_as_u8)
                .unwrap();
        }

        if let Some(debug_messenger) = &mut self.debug_messenger {
            debug_messenger.begin_label(cmd_buffer, &format!("Graph [{}]", graph.name));
        }

        unsafe {
            // Initialize all resources for the current frame state
            let mut image_barriers = Vec::new();

            for render_graph_image in &graph_frame_state.images {
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(render_graph_image.image.raw())
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .build();
                image_barriers.push(image_barrier);
            }

            if !image_barriers.is_empty() {
                if let Some(debug_messenger) = &mut self.debug_messenger {
                    debug_messenger.begin_label(cmd_buffer, "Initialization");
                }

                device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_barriers,
                );

                if let Some(debug_messenger) = &mut self.debug_messenger {
                    debug_messenger.end_label(cmd_buffer);
                }
            }

            // We don't actually use this set on GFX, but it is available in the fragment shaders.
            // This causes warnings if we don't update the set here for GFX too.
            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout.raw(),
                1,
                &[graph_frame_state.descriptor_set],
                &[],
            );

            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout.raw(),
                1,
                &[graph_frame_state.descriptor_set],
                &[],
            );

            for (batch_id, batch) in graph.batches.iter().enumerate() {
                if let Some(debug_messenger) = &mut self.debug_messenger {
                    debug_messenger.begin_label(
                        cmd_buffer,
                        &format!("Batch [{}/{}]", batch_id + 1, graph.batches.len()),
                    );
                }

                for node_index in &batch.node_indices {
                    let node = &graph.nodes[*node_index];

                    if let Some(debug_messenger) = &mut self.debug_messenger {
                        debug_messenger.begin_label(cmd_buffer, &format!("Node [{}]", node.name));
                    }

                    device.cmd_bind_pipeline(
                        cmd_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        node.pipeline.raw(),
                    );

                    // The texture slot index is stored inside the ImGui texture id
                    let push_constant_1 = node.get_ref(0).index();
                    let push_constant_2 = node.get_ref(1).index();
                    let push_constant_3 = node.get_ref(2).index();
                    let push_constants: [u32; 4] =
                        [0, push_constant_1, push_constant_2, push_constant_3];
                    device.cmd_push_constants(
                        cmd_buffer,
                        self.pipeline_layout.raw(),
                        vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::COMPUTE,
                        0,
                        push_constants.align_to::<u8>().1,
                    );

                    match &node.dispatch {
                        RenderGraphDispatchParams::Direct(direct) => {
                            device.cmd_dispatch(
                                cmd_buffer,
                                direct.num_groups_x,
                                direct.num_groups_y,
                                direct.num_groups_z,
                            );
                        }
                        RenderGraphDispatchParams::Indirect(indirect) => {
                            let buffer =
                                &graph_frame_state.buffers[indirect.buffer.index() as usize];
                            device.cmd_dispatch_indirect(
                                cmd_buffer,
                                buffer.buffer.raw(),
                                indirect.buffer_offset as u64,
                            );
                        }
                    }

                    if let Some(debug_messenger) = &mut self.debug_messenger {
                        debug_messenger.end_label(cmd_buffer);
                    }
                }

                if let Some(debug_messenger) = &mut self.debug_messenger {
                    debug_messenger.end_label(cmd_buffer);
                }

                // We must always insert barriers in-between batches to ensure that they don't overlap execution
                if batch_id != (graph.batches.len() - 1) {
                    // TODO: Usage of VkMemoryBarrier seems to automatically trigger L2 flush/invalidation which will
                    //       prevent any interesting L2-based optimizations and might make indirect dispatches slower.
                    //       We should switch to explicit image/buffer barriers to prevent this from happening.
                    //
                    //       Also, the indirect bits in the barrier should be removed when possible to prevent unnecessary
                    //       PFP syncs. Indirect bits should only be necessary when the NEXT batch in the graph contains an
                    //       indirect dispatch. (To really take advantage of this, we'll need to support some sort of
                    //       "swapchain sized" direct dispatch concept.)
                    device.cmd_pipeline_barrier(
                        cmd_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::DRAW_INDIRECT,
                        vk::DependencyFlags::empty(),
                        &[vk::MemoryBarrier::builder()
                            .src_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ
                                    | vk::AccessFlags::SHADER_WRITE
                                    | vk::AccessFlags::INDIRECT_COMMAND_READ,
                            )
                            .build()],
                        &[],
                        &[],
                    );
                }
            }
        }

        if let Some(debug_messenger) = &mut self.debug_messenger {
            debug_messenger.end_label(cmd_buffer);
        }

        Ok(())
    }
}

struct ImguiRenderer {
    #[allow(dead_code)]
    font_atlas_image: VkImage,
    font_atlas_image_view: VkImageView,
}

impl ImguiRenderer {
    fn new(
        device: &VkDevice,
        allocator: Weak<vk_mem::Allocator>,
        context: &mut imgui::Context,
    ) -> Result<Self> {
        let font_atlas_image;
        let font_atlas_image_view;
        {
            let mut context_fonts = context.fonts();
            let font_atlas = context_fonts.build_alpha8_texture();

            font_atlas_image = VkImage::new(
                allocator.clone(),
                &ash::vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D {
                        width: font_atlas.width,
                        height: font_atlas.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .format(vk::Format::R8_UNORM)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .samples(vk::SampleCountFlags::TYPE_1),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::GpuOnly,
                    ..Default::default()
                },
            )?;
            font_atlas_image_view = VkImageView::new(
                device.raw(),
                &vk::ImageViewCreateInfo::builder()
                    .image(font_atlas_image.raw())
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8_UNORM)
                    .components(
                        vk::ComponentMapping::builder()
                            .r(vk::ComponentSwizzle::IDENTITY)
                            .g(vk::ComponentSwizzle::IDENTITY)
                            .b(vk::ComponentSwizzle::IDENTITY)
                            .a(vk::ComponentSwizzle::IDENTITY)
                            .build(),
                    )
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    ),
            )?;

            let cmd_pool = VkCommandPool::new(
                device.raw(),
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(device.graphics_queue_family_index() as u32),
            )?;
            let cmd_buffer = cmd_pool.allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)?;
            unsafe {
                let raw_device = device.raw().upgrade().unwrap();
                raw_device.begin_command_buffer(
                    cmd_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                )?;

                // TODO: It would be faster to upload this with the transfer queue, but it would significantly increase
                //       the complexity of the upload process here. Replace this with a more standardized resource
                //       upload process when it becomes available.
                raw_device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(font_atlas_image.raw())
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .build()],
                );

                let atlas_buffer_size =
                    ((font_atlas.width * font_atlas.height) as usize) * std::mem::size_of::<u8>();
                let atlas_buffer = VkBuffer::new(
                    allocator,
                    &ash::vk::BufferCreateInfo::builder()
                        .size(atlas_buffer_size as u64)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::CpuToGpu,
                        flags: vk_mem::AllocationCreateFlags::MAPPED,
                        ..Default::default()
                    },
                )?;

                let atlas_data_src = font_atlas.data.as_ptr();
                let atlas_data_dst = atlas_buffer.info().get_mapped_data();
                core::ptr::copy_nonoverlapping(atlas_data_src, atlas_data_dst, atlas_buffer_size);

                raw_device.cmd_copy_buffer_to_image(
                    cmd_buffer,
                    atlas_buffer.raw(),
                    font_atlas_image.raw(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::builder()
                        .buffer_offset(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width: font_atlas.width,
                            height: font_atlas.height,
                            depth: 1,
                        })
                        .build()],
                );

                raw_device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(font_atlas_image.raw())
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .build()],
                );

                raw_device.end_command_buffer(cmd_buffer)?;
                raw_device.queue_submit(
                    device.graphics_queue(),
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[cmd_buffer])
                        .build()],
                    vk::Fence::null(),
                )?;
                raw_device.queue_wait_idle(device.graphics_queue())?;
            }
        }

        context.fonts().tex_id = imgui::TextureId::from(IMGUI_FONT_TEXTURE_SLOT_INDEX as usize);

        Ok(ImguiRenderer {
            font_atlas_image,
            font_atlas_image_view,
        })
    }
}
