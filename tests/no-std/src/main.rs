#![no_std]
#![no_main]

use core::alloc::{GlobalAlloc, Layout};

// This binary verifies bare-metal codegen and linking; it is not intended to run.
struct LinkProbeAllocator;

unsafe impl GlobalAlloc for LinkProbeAllocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        core::ptr::null_mut()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static ALLOCATOR: LinkProbeAllocator = LinkProbeAllocator;

#[no_mangle]
pub extern "C" fn _start() -> ! {
    core::hint::black_box(small_map_no_std_smoke::exercise_default_map());
    core::hint::black_box(small_map_no_std_smoke::exercise_default_set());
    core::hint::black_box(small_map_no_std_smoke::exercise_map());
    core::hint::black_box(small_map_no_std_smoke::exercise_set());

    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
