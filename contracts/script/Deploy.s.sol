// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "lib/openzeppelin-contracts/lib/forge-std/src/Script.sol";
import "../src/OracleSubscription.sol";
import "../src/UnifiedPredictionSubscription.sol";

contract DeployAll is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address treasury = vm.envOr("TREASURY_ADDRESS", address(0xdead));

        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy OracleSubscription (standalone)
        OracleSubscription oracle = new OracleSubscription(treasury);

        // 2. Deploy UnifiedPredictionSubscription (inherits OracleSubscription)
        //    CRE workflow author = deployer, workflow name = "agentswarm"
        address deployer = vm.addr(deployerPrivateKey);
        bytes10 workflowName = bytes10("agentswarm");
        UnifiedPredictionSubscription unified = new UnifiedPredictionSubscription(
            treasury,
            deployer,
            workflowName
        );

        // 3. Configure: set CRE automation address to deployer (for demo)
        oracle.setCREAutomation(deployer);

        // 4. Create a demo prediction market
        unified.createMarket(
            "Will ETH reach $5,000 by March 2026?",
            block.timestamp + 7 days
        );

        vm.stopBroadcast();

        // Log deployed addresses
        console.log("=== Deployment Complete ===");
        console.log("OracleSubscription:", address(oracle));
        console.log("UnifiedPredictionSubscription:", address(unified));
        console.log("Treasury:", treasury);
        console.log("CRE Workflow Author:", deployer);
        console.log("Workflow Name: agentswarm");
        console.log("Demo Market ID: 0");
    }
}
