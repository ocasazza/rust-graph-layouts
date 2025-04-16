{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust and WebAssembly tools
    rustup
    wasm-pack
    wasm-bindgen-cli
    
    # C/C++ toolchain
    clang
    llvmPackages.bintools
    
    # Other dependencies
    pkg-config
    openssl.dev
  ];
  
  shellHook = ''
    export PATH=$PATH:$HOME/.cargo/bin
    export RUSTFLAGS="-C link-arg=-fuse-ld=lld"
  '';
}
