`timescale 1ns/1ps
`default_nettype none
// Read-only weight memory, inferred as M10K block RAM.
// Registered read: data is valid ONE cycle after addr.
// INIT_FILE is resolved relative to where the tool runs: the repo root for Icarus,
// hw/quartus/ for Quartus (so pass "../../test_vectors/w1.hex" from the top level).
// INIT_FILE is a plain parameter, not "parameter string": Quartus rejects a
// string-typed parameter in $readmemh (Error 10686).
module weight_rom #(
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH = 147456,
    parameter INIT_FILE = "test_vectors/w1.hex"
)
(
    input logic                          clk,
    input logic [$clog2(DEPTH)-1:0]      addr,

    output logic signed [DATA_WIDTH-1:0] data
);
    logic signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    initial $readmemh(INIT_FILE, mem);

    always_ff @(posedge clk) begin
        data <= mem[addr];
    end
endmodule

`default_nettype wire
