use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use ash::{version::DeviceV1_0, vk};

use vkutil::*;

use crate::renderer::Renderer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct RenderGraphBufferParams {
    pub size: usize,
}

pub struct RenderGraphImageParams {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
}

pub enum RenderGraphResourceParams {
    // TODO: Implement buffer resources
    #[allow(dead_code)]
    Buffer(RenderGraphBufferParams),
    Image(RenderGraphImageParams),
}

pub enum RenderGraphPipelineSource<'a> {
    File(String),
    #[allow(dead_code)]
    Buffer(&'a [u32]),
}

pub struct RenderGraphNodeDesc<'a> {
    pub name: String,
    pub pipeline: RenderGraphPipelineSource<'a>,
    /// Referenced by shader like uTextures[Inputs[0]] where Inputs = push constants array
    pub refs: Vec<String>,
    pub dims: RenderGraphDispatchDimensions,
    pub deps: Vec<String>,
}

#[derive(Clone, Copy)]
pub struct RenderGraphDispatchDimensions {
    pub num_groups_x: u32,
    pub num_groups_y: u32,
    pub num_groups_z: u32,
}

pub struct RenderGraphResourceDesc {
    pub name: String,
    pub params: RenderGraphResourceParams,
}

// TODO: Replace Vecs with slices
pub struct RenderGraphDesc<'a> {
    pub resources: Vec<RenderGraphResourceDesc>,
    pub nodes: Vec<RenderGraphNodeDesc<'a>>,
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

pub struct RenderGraphNode {
    pub pipeline: VkPipeline,
    pub refs: Vec<ResourceReference>,
    pub dims: RenderGraphDispatchDimensions,
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

pub enum RenderGraphResource {
    Image(RenderGraphImage),
    Buffer(RenderGraphBuffer),
}

pub struct RenderGraphFrameState {
    pub resources: Vec<RenderGraphResource>,
    pub descriptor_set: vk::DescriptorSet,
}

pub struct RenderGraphBatch {
    pub node_indices: Vec<usize>,
}

pub struct RenderGraph {
    pub nodes: Vec<RenderGraphNode>,
    pub batches: Vec<RenderGraphBatch>,
    /// Used for lifetime purposes
    #[allow(dead_code)]
    descriptor_pool: VkDescriptorPool,
    pub frame_states: Vec<RenderGraphFrameState>,
    output_image_idx: Option<usize>,
}

const NUM_IMAGE_SLOTS: u32 = 64;

impl RenderGraph {
    pub fn new(desc: &RenderGraphDesc, renderer: &mut Renderer) -> Result<Self> {
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
                .pool_sizes(&[vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(num_frame_states * (NUM_IMAGE_SLOTS as u32))
                    .build()]),
        )?;

        let mut frame_states = Vec::new();
        for _frame_state_index in 0..num_frame_states {
            let descriptor_set = descriptor_pool
                .allocate_descriptor_set(renderer.get_graph_descriptor_set_layout().raw())?;

            let mut resources = Vec::new();
            let mut buffer_descs = Vec::new();
            let mut image_descs = Vec::new();

            for resource_desc in &desc.resources {
                match &resource_desc.params {
                    RenderGraphResourceParams::Buffer(params) => {
                        let buffer_resource = RenderGraphBuffer {
                            buffer: VkBuffer::new(
                                allocator.clone(),
                                &vk::BufferCreateInfo::builder()
                                    .size(params.size as u64)
                                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
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

                        let resource = RenderGraphResource::Buffer(buffer_resource);
                        resources.push(resource);
                    }
                    RenderGraphResourceParams::Image(params) => {
                        let vk_image = VkImage::new(
                            allocator.clone(),
                            &vk::ImageCreateInfo::builder()
                                .image_type(vk::ImageType::TYPE_2D)
                                .extent(vk::Extent3D {
                                    width: params.width,
                                    height: params.height,
                                    depth: 1,
                                })
                                .mip_levels(1)
                                .array_layers(1)
                                .format(params.format)
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
                                .format(params.format)
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

                        let resource = RenderGraphResource::Image(image_resource);
                        resources.push(resource);
                    }
                }
            }

            if image_descs.len() < NUM_IMAGE_SLOTS as usize {
                let last_index = image_descs.len() - 1;
                for _ in 0..(NUM_IMAGE_SLOTS as usize - image_descs.len()) {
                    image_descs.push(image_descs[last_index]);
                }
            }

            // TODO: Write Buffer Descs

            let image_desc_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&image_descs)
                .build();

            unsafe {
                device
                    .upgrade()
                    .unwrap()
                    .update_descriptor_sets(&[image_desc_write], &[]);
            }

            frame_states.push(RenderGraphFrameState {
                resources,
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
        }

        let mut node_mapping = HashMap::new();
        let mut nodes = Vec::new();
        for node_desc in &desc.nodes {
            let pipeline = match &node_desc.pipeline {
                RenderGraphPipelineSource::File(file_path) => {
                    renderer.create_compute_pipeline_from_file(file_path)?
                }
                RenderGraphPipelineSource::Buffer(buffer) => {
                    renderer.create_compute_pipeline_from_buffer(buffer)?
                }
            };
            let refs = node_desc
                .refs
                .iter()
                .map(|name| *resource_mapping.get(name).unwrap())
                .collect();
            let dims = node_desc.dims;
            let node = RenderGraphNode {
                pipeline,
                refs,
                dims,
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
            nodes,
            batches,
            descriptor_pool,
            frame_states,
            output_image_idx,
        })
    }

    pub fn get_output_image(&self, cur_frame_idx: usize) -> Option<&VkImageView> {
        if let Some(idx) = self.output_image_idx {
            if let RenderGraphResource::Image(image) =
                &self.frame_states[cur_frame_idx].resources[idx]
            {
                return Some(&image.view);
            }
        }
        None
    }
}
