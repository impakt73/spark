use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use ash::{version::DeviceV1_0, vk};

use vkutil::*;

use serde::{Deserialize, Serialize};

use crate::renderer::Renderer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Serialize, Deserialize, Debug)]
pub struct RenderGraphBufferParams {
    pub size: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RenderGraphFixedImageParams {
    pub width: u32,
    pub height: u32,
    pub format: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RenderGraphImageParams {
    SwapchainCompatible,
    Fixed(RenderGraphFixedImageParams),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RenderGraphResourceParams {
    Buffer(RenderGraphBufferParams),
    Image(RenderGraphImageParams),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RenderGraphPipelineSource {
    File(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RenderGraphNodeDesc {
    pub name: String,
    pub pipeline: RenderGraphPipelineSource,
    /// Referenced by shader like uTextures[Inputs[0]] where Inputs = push constants array
    pub refs: Vec<String>,
    pub dispatch: RenderGraphDispatchDesc,
    pub deps: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RenderGraphDirectDispatchDesc {
    pub num_groups_x: u32,
    pub num_groups_y: u32,
    pub num_groups_z: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RenderGraphIndirectDispatchDesc {
    pub buffer: String,
    pub offset: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RenderGraphDispatchDesc {
    Direct(RenderGraphDirectDispatchDesc),
    Indirect(RenderGraphIndirectDispatchDesc),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RenderGraphResourceDesc {
    pub name: String,
    pub params: RenderGraphResourceParams,
}

// TODO: Replace Vecs with slices
#[derive(Serialize, Deserialize, Debug)]
pub struct RenderGraphDesc {
    pub name: String,
    pub resources: Vec<RenderGraphResourceDesc>,
    pub nodes: Vec<RenderGraphNodeDesc>,
    pub output_image_name: Option<String>,
}

#[derive(Clone, Copy)]
pub enum ResourceReference {
    Buffer { index: u32 },
    Image { index: u32 },
    Invalid,
}

impl ResourceReference {
    pub fn index(&self) -> u32 {
        match self {
            ResourceReference::Buffer { index } => *index,
            ResourceReference::Image { index } => *index,
            ResourceReference::Invalid => !0,
        }
    }
}

impl Default for ResourceReference {
    fn default() -> Self {
        ResourceReference::Invalid
    }
}

pub struct RenderGraphDirectDispatchParams {
    pub num_groups_x: u32,
    pub num_groups_y: u32,
    pub num_groups_z: u32,
}

pub struct RenderGraphIndirectDispatchParams {
    pub buffer: ResourceReference,
    pub buffer_offset: usize,
}

pub enum RenderGraphDispatchParams {
    Direct(RenderGraphDirectDispatchParams),
    Indirect(RenderGraphIndirectDispatchParams),
}

pub struct RenderGraphNode {
    pub name: String,
    pub pipeline: VkPipeline,
    pub refs: Vec<ResourceReference>,
    pub dispatch: RenderGraphDispatchParams,
}

impl RenderGraphNode {
    pub fn get_ref(&self, slot_idx: usize) -> ResourceReference {
        if let Some(resource_ref) = self.refs.get(slot_idx) {
            *resource_ref
        } else {
            ResourceReference::Invalid
        }
    }
}

pub struct RenderGraphImage {
    pub view: VkImageView,
    pub image: VkImage,
}

pub struct RenderGraphBuffer {
    pub buffer: VkBuffer,
}

pub struct RenderGraphFrameState {
    pub buffers: Vec<RenderGraphBuffer>,
    pub images: Vec<RenderGraphImage>,
    pub descriptor_set: vk::DescriptorSet,
}

pub struct RenderGraphBatch {
    pub node_indices: Vec<usize>,
}

/// Attempts to convert a string into a VkFormat enum
fn vk_format_from_string(string: &str) -> vk::Format {
    match string {
        "rgba8" => vk::Format::R8G8B8A8_UNORM,
        "r32" => vk::Format::R32_UINT,
        _ => vk::Format::UNDEFINED,
    }
}

pub struct RenderGraph {
    pub name: String,
    pub nodes: Vec<RenderGraphNode>,
    pub batches: Vec<RenderGraphBatch>,
    /// Used for lifetime purposes
    #[allow(dead_code)]
    descriptor_pool: VkDescriptorPool,
    pub frame_states: Vec<RenderGraphFrameState>,
    output_image_idx: Option<usize>,
}

const NUM_IMAGE_SLOTS: u32 = 64;
const NUM_BUFFER_SLOTS: u32 = 64;

impl RenderGraph {
    pub fn new(
        desc: &RenderGraphDesc,
        resource_dir: Option<&str>,
        renderer: &mut Renderer,
    ) -> Result<Self> {
        let device = Arc::downgrade(&renderer.get_device());
        let allocator = renderer.get_allocator();
        let num_frame_states = renderer.get_num_frame_states() as u32;

        let mut resource_mapping = HashMap::new();
        let mut buffer_count = 0;
        let mut image_count = 0;

        // Allocate slots for all resources
        for resource_desc in &desc.resources {
            let resource_name = resource_desc.name.clone();

            match &resource_desc.params {
                RenderGraphResourceParams::Buffer(_) => {
                    let reference = ResourceReference::Buffer {
                        index: buffer_count,
                    };
                    resource_mapping.insert(resource_name, reference);
                    buffer_count += 1;
                }
                RenderGraphResourceParams::Image(_) => {
                    let reference = ResourceReference::Image { index: image_count };
                    resource_mapping.insert(resource_name, reference);
                    image_count += 1;
                }
            }
        }

        let descriptor_pool = VkDescriptorPool::new(
            device.clone(),
            &vk::DescriptorPoolCreateInfo::builder()
                .max_sets(num_frame_states)
                .pool_sizes(&[
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(num_frame_states * (NUM_IMAGE_SLOTS as u32))
                        .build(),
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(num_frame_states * (NUM_BUFFER_SLOTS as u32))
                        .build(),
                ]),
        )?;

        let mut frame_states = Vec::new();
        for _frame_state_index in 0..num_frame_states {
            let descriptor_set = descriptor_pool
                .allocate_descriptor_set(renderer.get_graph_descriptor_set_layout().raw())?;

            let mut buffers = Vec::new();
            let mut images = Vec::new();
            let mut buffer_descs = Vec::new();
            let mut image_descs = Vec::new();

            let (swapchain_width, swapchain_height) = renderer.get_swapchain_resolution();
            let swapchain_format = renderer.get_swapchain_format();

            for resource_desc in &desc.resources {
                match &resource_desc.params {
                    RenderGraphResourceParams::Buffer(params) => {
                        let buffer_resource = RenderGraphBuffer {
                            buffer: VkBuffer::new(
                                allocator.clone(),
                                &vk::BufferCreateInfo::builder()
                                    .size(params.size as u64)
                                    .usage(
                                        vk::BufferUsageFlags::STORAGE_BUFFER
                                            | vk::BufferUsageFlags::INDIRECT_BUFFER,
                                    ),
                                &vk_mem::AllocationCreateInfo {
                                    usage: vk_mem::MemoryUsage::GpuOnly,
                                    ..Default::default()
                                },
                            )?,
                        };

                        let desc = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer_resource.buffer.raw())
                            .offset(0)
                            .range(params.size as u64)
                            .build();
                        buffer_descs.push(desc);

                        buffers.push(buffer_resource);
                    }
                    RenderGraphResourceParams::Image(params) => {
                        let image_extent = match &params {
                            RenderGraphImageParams::SwapchainCompatible => vk::Extent3D {
                                width: swapchain_width,
                                height: swapchain_height,
                                depth: 1,
                            },
                            RenderGraphImageParams::Fixed(fixed_params) => vk::Extent3D {
                                width: fixed_params.width,
                                height: fixed_params.height,
                                depth: 1,
                            },
                        };
                        let image_format = match &params {
                            RenderGraphImageParams::SwapchainCompatible => swapchain_format,
                            RenderGraphImageParams::Fixed(fixed_params) => {
                                vk_format_from_string(&fixed_params.format)
                            }
                        };
                        let vk_image = VkImage::new(
                            allocator.clone(),
                            &vk::ImageCreateInfo::builder()
                                .image_type(vk::ImageType::TYPE_2D)
                                .extent(image_extent)
                                .mip_levels(1)
                                .array_layers(1)
                                .format(image_format)
                                .tiling(vk::ImageTiling::OPTIMAL)
                                .initial_layout(vk::ImageLayout::UNDEFINED)
                                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                .samples(vk::SampleCountFlags::TYPE_1),
                            &vk_mem::AllocationCreateInfo {
                                usage: vk_mem::MemoryUsage::GpuOnly,
                                ..Default::default()
                            },
                        )?;
                        let vk_image_view = VkImageView::new(
                            device.clone(),
                            &vk::ImageViewCreateInfo::builder()
                                .image(vk_image.raw())
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(image_format)
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
                        let image_resource = RenderGraphImage {
                            view: vk_image_view,
                            image: vk_image,
                        };

                        let desc = vk::DescriptorImageInfo::builder()
                            .image_view(image_resource.view.raw())
                            .image_layout(vk::ImageLayout::GENERAL)
                            .build();
                        image_descs.push(desc);

                        images.push(image_resource);
                    }
                }
            }

            let mut desc_writes = Vec::new();

            if !image_descs.is_empty() {
                if image_descs.len() < NUM_IMAGE_SLOTS as usize {
                    let last_index = image_descs.len() - 1;
                    for _ in 0..(NUM_IMAGE_SLOTS as usize - image_descs.len()) {
                        image_descs.push(image_descs[last_index]);
                    }
                }

                let image_desc_write = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&image_descs)
                    .build();

                desc_writes.push(image_desc_write);
            }

            if !buffer_descs.is_empty() {
                if buffer_descs.len() < NUM_BUFFER_SLOTS as usize {
                    let last_index = buffer_descs.len() - 1;
                    for _ in 0..(NUM_BUFFER_SLOTS as usize - buffer_descs.len()) {
                        buffer_descs.push(buffer_descs[last_index]);
                    }
                }

                let buffer_desc_write = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_descs)
                    .build();

                desc_writes.push(buffer_desc_write);
            }

            if !desc_writes.is_empty() {
                unsafe {
                    device
                        .upgrade()
                        .unwrap()
                        .update_descriptor_sets(&desc_writes, &[]);
                }
            }

            frame_states.push(RenderGraphFrameState {
                buffers,
                images,
                descriptor_set,
            });
        }

        // Validate all resource references before we attempt to create real nodes
        for node in &desc.nodes {
            for resource_name in &node.refs {
                resource_mapping
                    .get(resource_name)
                    .copied()
                    .ok_or("invalid input reference")?;
            }

            if let RenderGraphDispatchDesc::Indirect(dispatch_desc) = &node.dispatch {
                resource_mapping
                    .get(&dispatch_desc.buffer)
                    .copied()
                    .ok_or("Invalid dispatch indirect buffer")?;
            }
        }

        let mut node_mapping = HashMap::new();
        let mut nodes = Vec::new();
        for node_desc in &desc.nodes {
            let pipeline = match &node_desc.pipeline {
                RenderGraphPipelineSource::File(filename) => {
                    let path = if let Some(res_dir_path) = resource_dir {
                        Path::new(res_dir_path).join(filename)
                    } else {
                        PathBuf::from(filename)
                    };
                    renderer.create_compute_pipeline_from_file(path.to_str().unwrap())?
                }
            };
            let refs = node_desc
                .refs
                .iter()
                .map(|name| *resource_mapping.get(name).unwrap())
                .collect();

            let dispatch = match &node_desc.dispatch {
                RenderGraphDispatchDesc::Direct(dispatch_desc) => {
                    RenderGraphDispatchParams::Direct(RenderGraphDirectDispatchParams {
                        num_groups_x: dispatch_desc.num_groups_x,
                        num_groups_y: dispatch_desc.num_groups_y,
                        num_groups_z: dispatch_desc.num_groups_z,
                    })
                }
                RenderGraphDispatchDesc::Indirect(dispatch_desc) => {
                    let buffer = *resource_mapping.get(&dispatch_desc.buffer).unwrap();
                    RenderGraphDispatchParams::Indirect(RenderGraphIndirectDispatchParams {
                        buffer,
                        buffer_offset: dispatch_desc.offset,
                    })
                }
            };

            let node = RenderGraphNode {
                name: node_desc.name.clone(),
                pipeline,
                refs,
                dispatch,
            };

            // Track the index of each render graph node to help with dependency lookup later
            node_mapping.insert(node_desc.name.clone(), nodes.len());

            nodes.push(node);
        }

        // Compute node batches
        let mut batches = Vec::new();
        let mut executed_nodes = HashSet::new();
        let mut nodes_to_execute = desc.nodes.iter().enumerate().collect::<Vec<_>>();
        while !nodes_to_execute.is_empty() {
            let mut node_indices = Vec::new();
            let num_nodes_to_execute = nodes_to_execute.len();
            nodes_to_execute.retain(|(node_index, node_desc)| {
                let mut is_ready = false;

                if node_desc.deps.is_empty() {
                    is_ready = true;
                } else {
                    let mut all_deps_met = true;
                    for dep in &node_desc.deps {
                        if !executed_nodes.contains(dep) {
                            all_deps_met = false;
                            break;
                        }
                    }
                    if all_deps_met {
                        is_ready = true;
                    }
                }

                if is_ready {
                    node_indices.push(*node_index);
                }

                !is_ready
            });

            // Mark the all nodes in the current batch as executed
            for node_index in &node_indices {
                executed_nodes.insert(desc.nodes[*node_index].name.clone());
            }

            batches.push(RenderGraphBatch { node_indices });

            if num_nodes_to_execute == nodes_to_execute.len() {
                return Err("Unable to satisfy render graph node dependencies!".into());
            }
        }

        let mut output_image_idx = None;

        if let Some(output_image_name) = &desc.output_image_name {
            if let Some(output_image_ref) = resource_mapping.get(output_image_name) {
                output_image_idx = Some(output_image_ref.index() as usize);
            }
        }

        Ok(Self {
            name: desc.name.clone(),
            nodes,
            batches,
            descriptor_pool,
            frame_states,
            output_image_idx,
        })
    }

    pub fn get_output_image(&self, cur_frame_idx: usize) -> Option<&VkImageView> {
        if let Some(idx) = self.output_image_idx {
            Some(&self.frame_states[cur_frame_idx].images[idx].view)
        } else {
            None
        }
    }
}
