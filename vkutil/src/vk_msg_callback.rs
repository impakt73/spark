use ash::vk;

use crate::log::*;

use std::ffi::CStr;

// We can't use a runtime value for the Level option, so switch and call each macro directly
// This lets us avoid writing our format string five times.
macro_rules! vk_log {
        ($level:expr, $($rest:tt)+) => {
            match $level {
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => error!($($rest)+),
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!($($rest)+),
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO => info!($($rest)+),
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => trace!($($rest)+),
                _ => {
                    error!($($rest)+);
                    warn!("Unrecognized severity: {:?}", $level)
                }
            };
        };
    }

/// Extremely unsafe-in-general function to ease making slices from Vulkan structs
///
/// To make sure it matches a reasonable lifetime, it derives it from the Count field.
unsafe fn unsafe_slice<T>(t_count: &u32, p_ts: *const T) -> &[T] {
    if *t_count != 0 {
        std::slice::from_raw_parts(p_ts, *t_count as usize)
    } else {
        &[]
    }
}

/// Callback for `VK_EXT_debug_utils`.
pub unsafe extern "system" fn vulkan_debug_messenger_cb(
    msg_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut core::ffi::c_void,
) -> vk::Bool32 {
    // Start a span to record how much time we spend in these
    const SPAN_NAME: &str = "Vk Debug Msg Callback";
    let span = match msg_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => error_span!(SPAN_NAME),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn_span!(SPAN_NAME),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => info_span!(SPAN_NAME),
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => trace_span!(SPAN_NAME),
        // Unknown levels should be logged as an error
        _ => error_span!(SPAN_NAME),
    };
    let _span_enter = span.enter();

    let is_general = msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL);
    let is_validation = msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION);
    let is_performance = msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE);

    vk_log!(
        msg_severity, target: "Vulkan",
        ?is_general,
        ?is_validation,
        ?is_performance,
    );

    let vk::DebugUtilsMessengerCallbackDataEXT {
        flags,

        p_message_id_name: _,
        message_id_number: _,
        p_message,

        queue_label_count,
        p_queue_labels,

        cmd_buf_label_count,
        p_cmd_buf_labels,

        object_count,
        p_objects,
        ..
    } = *cb_data
        .as_ref()
        .expect("DebugUtilsMessengerCallbackDataEXT was NULL");

    // `flags` is currently reserved for future use. Add a check for if that ever happens
    #[cfg(debug_assertions)]
    if !flags.is_empty() {
        warn!(
            concat!(
                "VkDebugUtilsMessengerCallbackDataFlagsEXT now has flags: 0b{:b}.",
                "But we don't know what they mean! 🤷‍♀️ You may want to update Ash."
            ),
            flags.as_raw()
        );
    }

    let _queues: &[vk::DebugUtilsLabelEXT] = unsafe_slice(&queue_label_count, p_queue_labels);
    let _cmdbufs: &[vk::DebugUtilsLabelEXT] = unsafe_slice(&cmd_buf_label_count, p_cmd_buf_labels);
    let objs: &[vk::DebugUtilsObjectNameInfoEXT] = unsafe_slice(&object_count, p_objects);

    for obj in objs {
        let vk::DebugUtilsObjectNameInfoEXT {
            p_next,
            object_type: type_,
            object_handle: handle,
            p_object_name: name,
            ..
        } = *obj;

        let name = if name.is_null() {
            ""
        } else {
            CStr::from_ptr(name).to_str().unwrap_or("<invalid UTF8>")
        };

        vk_log!(msg_severity, target: "Vulkan", name, ?type_, ?handle, ?p_next);
    }

    // This is the full message for the callback. It's often long and should be broken across multiple lines.
    let full_message = CStr::from_ptr(p_message)
        .to_str()
        .unwrap_or("<invalid UTF8>");

    vk_log!(msg_severity, target: "Vulkan", "{}", full_message);

    // From the Vulkan spec:
    //      The callback returns a VkBool32, which is interpreted in a layer-specified manner.
    //      The application should always return VK_FALSE. The VK_TRUE value is reserved for use
    //      in layer development.
    vk::FALSE
}
