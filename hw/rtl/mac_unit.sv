module mac_unit#(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)
(
    input logic                          clk,
    input logic                          en,
    input logic                          lb,
    input logic                          rst_n,
    input logic signed [DATA_WIDTH-1:0]  W1_q,
    input logic signed [DATA_WIDTH-1:0]  w_q,
    input logic signed [ACC_WIDTH-1:0]   b1_q,

    output logic signed [ACC_WIDTH-1:0] acc1
);
    logic signed [ACC_WIDTH-1:0] w_result;
    logic signed [ACC_WIDTH-1:0] next_acc;

    assign w_result = W1_q * w_q;
    assign next_acc = acc1 + w_result;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc1 <= 0;
        end else if (en) begin
            if (lb) begin
                acc1 <= b1_q;
            end else begin
                acc1 <= next_acc;
            end
        end
    end
endmodule
