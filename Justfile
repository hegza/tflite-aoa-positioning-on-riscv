deploy:
    cargo build
    cp target/riscv32imac-unknown-none-elf/debug/rust-proto-4 ../renode-riscv/binaries/demo
