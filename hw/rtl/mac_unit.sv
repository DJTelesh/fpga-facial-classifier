`timescale 1ns/1ps

// One multiply-accumulate lane. Layer-agnostic: layer 1 feeds it (W1, x_q),
// layer 2 feeds it (W2, h_q).
//   lb=1: acc <= bias            (starts a new neuron, discards the old sum)
//   lb=0: acc <= acc + w * x
// en=0 holds acc, and takes priority over lb.
module mac_unit#(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)
(
    input logic                          clk,
    input logic                          en,
    input logic                          lb,
    input logic                          rst_n,
    input logic signed [DATA_WIDTH-1:0]  w,     // weight
    input logic signed [DATA_WIDTH-1:0]  x,     // input / activation
    input logic signed [ACC_WIDTH-1:0]   bias,

    output logic signed [ACC_WIDTH-1:0] acc
);
    logic signed [ACC_WIDTH-1:0] w_result;
    logic signed [ACC_WIDTH-1:0] next_acc;

    assign w_result = w * x;
    assign next_acc = acc + w_result;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc <= 0;
        end else if (en) begin
            if (lb) begin
                acc <= bias;
            end else begin
                acc <= next_acc;
            end
        end
    end
endmodule
