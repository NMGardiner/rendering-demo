use std::{
    error::Error,
    ffi::{c_void, CStr, CString},
    ops::Deref,
};

use raw_window_handle::HasRawWindowHandle;

/// An object for building renderer instances.
pub struct InstanceBuilder<'a> {
    application_name: String,
    application_version: u32,
    window_handle: Option<&'a dyn HasRawWindowHandle>,
    layers: Vec<CString>,
    extensions: Vec<&'a CStr>,
}

impl<'a> InstanceBuilder<'a> {
    /// Set the name of the application.
    pub fn application_name(mut self, application_name: &str) -> Self {
        self.application_name = String::from(application_name);
        self
    }

    /// Set the version of the application.
    pub fn application_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.application_version = ash::vk::make_api_version(0, major, minor, patch);
        self
    }

    /// Bind a window handle to the renderer instance. The necessary extensions will be enabled - do not enable
    /// them manually.
    pub fn window_handle(mut self, window_handle: &'a dyn HasRawWindowHandle) -> Self {
        self.window_handle = Some(window_handle);
        self
    }

    /// Add the given layer names to be enabled.
    pub fn with_layers(mut self, layers: &[&str]) -> Self {
        self.layers.extend(
            layers
                .iter()
                .map(|&layer| CString::new(layer).unwrap_or_default()),
        );
        self
    }

    /// Add the given extension names to be enabled.
    pub fn with_extensions(mut self, extensions: &'a [&'a CStr]) -> Self {
        self.extensions.extend(extensions);
        self
    }

    /// Build the renderer instance.
    ///
    /// See [`Instance::new`] for details.
    pub fn build(self) -> Result<Instance, Box<dyn Error>> {
        Instance::new(self)
    }
}

impl<'a> Default for InstanceBuilder<'a> {
    fn default() -> Self {
        Self {
            application_name: String::from(""),
            application_version: ash::vk::make_api_version(0, 1, 0, 0),
            window_handle: None,
            layers: vec![],
            extensions: vec![],
        }
    }
}

// The entry must outlive the instance, but isn't necessarily used.
#[allow(dead_code)]
/// A renderer instance, storing the `ash` entry, instance, and debug messenger (if applicable).
pub struct Instance {
    entry: ash::Entry,
    instance_handle: ash::Instance,
    debug_messenger: Option<DebugMessenger>,
}

impl Instance {
    /// Create an [`InstanceBuilder`] to build a renderer instance..
    pub fn builder() -> InstanceBuilder<'static> {
        InstanceBuilder::default()
    }

    /// Create a new renderer instance from the given builder's parameters.
    ///
    /// # Errors
    ///
    /// This function can error if `ash` fails to create the entry, instance, or debug messenger,
    /// if `ash` fails to enumerate the supported instance layers/extensions, or if `ash_window`
    /// fails to enumerate the required window extensions.
    pub fn new(mut builder: InstanceBuilder) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { ash::Entry::load()? };

        // Catch any nul bytes in the given application string.
        let application_name = match CString::new(builder.application_name) {
            Ok(name) => name,
            Err(error) => {
                log::warn!(
                    "Invalid application name given: {}. Falling back to an empty string.",
                    error.to_string()
                );

                CString::new("")?
            }
        };

        let application_info = ash::vk::ApplicationInfo::builder()
            .application_name(&application_name)
            .application_version(builder.application_version)
            .engine_name(CStr::from_bytes_with_nul(b"Vulkan Engine\0")?)
            .engine_version(ash::vk::make_api_version(0, 0, 1, 0))
            .api_version(ash::vk::API_VERSION_1_3);

        // Keep only the supported layers to enable.
        drop_unsupported_layers(&entry, &mut builder.layers)?;

        // The InstanceCreateInfo needs the layer and extension names as a *const c_char slice.
        let enabled_layers = builder
            .layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<Vec<_>>();

        // If there's a window attached, add the required window extensions.
        if let Some(window_handle) = builder.window_handle {
            builder.extensions.append(
                ash_window::enumerate_required_extensions(window_handle)?
                    .iter()
                    .map(|&extension_name| unsafe { CStr::from_ptr(extension_name) })
                    .collect::<Vec<_>>()
                    .as_mut(),
            )
        }

        if !verify_extension_support(&entry, &builder.extensions)? {
            return Err("One or more required instance extensions are unsupported. Initialisation can not continue.".to_string().into());
        }

        // The InstanceCreateInfo needs the layer and extension names as a *const c_char slice.
        let enabled_extensions = builder
            .extensions
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<Vec<_>>();

        let mut instance_info = ash::vk::InstanceCreateInfo::builder()
            .flags(ash::vk::InstanceCreateFlags::empty())
            .application_info(&application_info)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_extensions);

        let mut debug_messenger_info = ash::vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .flags(ash::vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(
                ash::vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                ash::vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | ash::vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | ash::vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback))
            .user_data(std::ptr::null_mut());

        let create_debug_messenger = builder
            .extensions
            .contains(&ash::extensions::ext::DebugUtils::name());

        if create_debug_messenger {
            instance_info = instance_info.push_next(&mut debug_messenger_info);
        }

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let debug_messenger = if create_debug_messenger {
            Some(DebugMessenger::new(
                &entry,
                &instance,
                &mut debug_messenger_info,
            )?)
        } else {
            None
        };

        Ok(Self {
            entry,
            instance_handle: instance,
            debug_messenger,
        })
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn handle(&self) -> &ash::Instance {
        &self.instance_handle
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.debug_messenger = None;
            self.instance_handle.destroy_instance(None);
        }
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        self.handle()
    }
}

fn drop_unsupported_layers(
    entry: &ash::Entry,
    requested_layers: &mut Vec<CString>,
) -> Result<(), Box<dyn Error>> {
    let supported_layer_properties = entry.enumerate_instance_layer_properties()?;

    let supported_layer_names = supported_layer_properties
        .iter()
        .map(|supported_layer| unsafe { CStr::from_ptr(supported_layer.layer_name.as_ptr()) })
        .collect::<Vec<_>>();

    requested_layers.retain(|layer_name| {
        if supported_layer_names.contains(&layer_name.as_c_str()) {
            true
        } else {
            let unsupported_layer_name = match layer_name.to_str() {
                Ok(name) => name,
                Err(_error) => "[INVALID_NAME]",
            };

            log::warn!(
                "The requested layer {} is unsupported and will not be enabled.",
                unsupported_layer_name
            );

            false
        }
    });

    Ok(())
}

fn verify_extension_support(
    entry: &ash::Entry,
    required_extensions: &[&CStr],
) -> Result<bool, Box<dyn Error>> {
    let supported_extension_properties = entry.enumerate_instance_extension_properties(None)?;

    let mut missing_any_extensions = false;
    for required_extension in required_extensions.iter() {
        if !supported_extension_properties
            .iter()
            .map(|supported_extension| unsafe {
                CStr::from_ptr(supported_extension.extension_name.as_ptr())
            })
            .any(|supported_extension| supported_extension == *required_extension)
        {
            let unsupported_extension_name = match required_extension.to_str() {
                Ok(name) => name,
                Err(_error) => "[INVALID_NAME]",
            };

            log::error!(
                "The required instance extension {} is unsupported.",
                unsupported_extension_name
            );
            missing_any_extensions = true;
        }
    }

    Ok(!missing_any_extensions)
}

/// A wrapper for `ash`'s DebugUtils loader and messenger.
struct DebugMessenger {
    debug_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: ash::vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        create_info: &mut ash::vk::DebugUtilsMessengerCreateInfoEXTBuilder,
    ) -> Result<Self, Box<dyn Error>> {
        let debug_loader = ash::extensions::ext::DebugUtils::new(entry, instance);

        let debug_messenger =
            unsafe { debug_loader.create_debug_utils_messenger(create_info, None)? };

        Ok(Self {
            debug_loader,
            debug_messenger,
        })
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        unsafe {
            self.debug_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
        }
    }
}

// Custom callback for the validation layers.
unsafe extern "system" fn debug_callback(
    message_flag: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> ash::vk::Bool32 {
    use ash::vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

    let message = CStr::from_ptr((*p_callback_data).p_message);
    match message_flag {
        Flag::INFO => log::info!("{:?} - {:?}", message_type, message),
        Flag::WARNING => log::warn!("{:?} - {:?}", message_type, message),
        Flag::ERROR => log::error!("{:?} - {:?}", message_type, message),
        _ => (),
    }

    ash::vk::FALSE
}
