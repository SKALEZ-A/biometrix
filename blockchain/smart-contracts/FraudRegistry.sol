// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title FraudRegistry
 * @dev Decentralized fraud case registry with privacy-preserving features
 * @notice Stores fraud cases on-chain with IPFS evidence links
 */
contract FraudRegistry is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant FRAUD_ANALYST_ROLE = keccak256("FRAUD_ANALYST_ROLE");
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");
    
    struct FraudCase {
        bytes32 caseId;
        address reporter;
        uint256 timestamp;
        uint256 amount;
        FraudType fraudType;
        Severity severity;
        CaseStatus status;
        string ipfsHash;  // IPFS hash of encrypted evidence
        bytes32 userIdHash;  // Hashed user ID for privacy
        bytes32 transactionIdHash;  // Hashed transaction ID
        uint256 resolutionTimestamp;
        string resolutionNotes;
        bool confirmed;
    }
    
    enum FraudType {
        AccountTakeover,
        SyntheticIdentity,
        PaymentFraud,
        Chargeback,
        MoneyLaundering,
        PhishingAttack,
        BotAttack,
        Other
    }
    
    enum Severity {
        Low,
        Medium,
        High,
        Critical
    }
    
    enum CaseStatus {
        Open,
        Investigating,
        Resolved,
        FalsePositive,
        Escalated
    }
    
    // Mapping from case ID to fraud case
    mapping(bytes32 => FraudCase) public fraudCases;
    
    // Mapping from user ID hash to their fraud cases
    mapping(bytes32 => bytes32[]) public userFraudCases;
    
    // Mapping from transaction ID hash to case ID
    mapping(bytes32 => bytes32) public transactionToCaseId;
    
    // Array of all case IDs
    bytes32[] public allCaseIds;
    
    // Statistics
    uint256 public totalCases;
    uint256 public confirmedFraudCases;
    uint256 public totalFraudAmount;
    
    // Events
    event FraudCaseCreated(
        bytes32 indexed caseId,
        address indexed reporter,
        FraudType fraudType,
        Severity severity,
        uint256 amount
    );
    
    event FraudCaseUpdated(
        bytes32 indexed caseId,
        CaseStatus newStatus,
        address updatedBy
    );
    
    event FraudCaseResolved(
        bytes32 indexed caseId,
        bool confirmed,
        uint256 resolutionTimestamp
    );
    
    event EvidenceAdded(
        bytes32 indexed caseId,
        string ipfsHash,
        address addedBy
    );
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(FRAUD_ANALYST_ROLE, msg.sender);
        _grantRole(AUDITOR_ROLE, msg.sender);
    }
    
    /**
     * @dev Create a new fraud case
     * @param _userIdHash Hashed user ID
     * @param _transactionIdHash Hashed transaction ID
     * @param _amount Transaction amount
     * @param _fraudType Type of fraud
     * @param _severity Severity level
     * @param _ipfsHash IPFS hash of encrypted evidence
     */
    function createFraudCase(
        bytes32 _userIdHash,
        bytes32 _transactionIdHash,
        uint256 _amount,
        FraudType _fraudType,
        Severity _severity,
        string memory _ipfsHash
    ) external onlyRole(FRAUD_ANALYST_ROLE) whenNotPaused returns (bytes32) {
        bytes32 caseId = keccak256(
            abi.encodePacked(
                _userIdHash,
                _transactionIdHash,
                block.timestamp,
                msg.sender
            )
        );
        
        require(fraudCases[caseId].timestamp == 0, "Case already exists");
        
        FraudCase memory newCase = FraudCase({
            caseId: caseId,
            reporter: msg.sender,
            timestamp: block.timestamp,
            amount: _amount,
            fraudType: _fraudType,
            severity: _severity,
            status: CaseStatus.Open,
            ipfsHash: _ipfsHash,
            userIdHash: _userIdHash,
            transactionIdHash: _transactionIdHash,
            resolutionTimestamp: 0,
            resolutionNotes: "",
            confirmed: false
        });
        
        fraudCases[caseId] = newCase;
        userFraudCases[_userIdHash].push(caseId);
        transactionToCaseId[_transactionIdHash] = caseId;
        allCaseIds.push(caseId);
        
        totalCases++;
        
        emit FraudCaseCreated(caseId, msg.sender, _fraudType, _severity, _amount);
        
        return caseId;
    }
    
    /**
     * @dev Update fraud case status
     * @param _caseId Case ID
     * @param _newStatus New status
     */
    function updateCaseStatus(
        bytes32 _caseId,
        CaseStatus _newStatus
    ) external onlyRole(FRAUD_ANALYST_ROLE) whenNotPaused {
        require(fraudCases[_caseId].timestamp != 0, "Case does not exist");
        
        fraudCases[_caseId].status = _newStatus;
        
        emit FraudCaseUpdated(_caseId, _newStatus, msg.sender);
    }
    
    /**
     * @dev Resolve fraud case
     * @param _caseId Case ID
     * @param _confirmed Whether fraud was confirmed
     * @param _resolutionNotes Resolution notes
     */
    function resolveFraudCase(
        bytes32 _caseId,
        bool _confirmed,
        string memory _resolutionNotes
    ) external onlyRole(FRAUD_ANALYST_ROLE) whenNotPaused {
        require(fraudCases[_caseId].timestamp != 0, "Case does not exist");
        require(fraudCases[_caseId].resolutionTimestamp == 0, "Case already resolved");
        
        fraudCases[_caseId].status = _confirmed ? CaseStatus.Resolved : CaseStatus.FalsePositive;
        fraudCases[_caseId].confirmed = _confirmed;
        fraudCases[_caseId].resolutionTimestamp = block.timestamp;
        fraudCases[_caseId].resolutionNotes = _resolutionNotes;
        
        if (_confirmed) {
            confirmedFraudCases++;
            totalFraudAmount += fraudCases[_caseId].amount;
        }
        
        emit FraudCaseResolved(_caseId, _confirmed, block.timestamp);
    }
    
    /**
     * @dev Add additional evidence to a case
     * @param _caseId Case ID
     * @param _ipfsHash IPFS hash of new evidence
     */
    function addEvidence(
        bytes32 _caseId,
        string memory _ipfsHash
    ) external onlyRole(FRAUD_ANALYST_ROLE) whenNotPaused {
        require(fraudCases[_caseId].timestamp != 0, "Case does not exist");
        
        // Append to existing IPFS hash (comma-separated)
        string memory currentHash = fraudCases[_caseId].ipfsHash;
        fraudCases[_caseId].ipfsHash = string(abi.encodePacked(currentHash, ",", _ipfsHash));
        
        emit EvidenceAdded(_caseId, _ipfsHash, msg.sender);
    }
    
    /**
     * @dev Get fraud case details
     * @param _caseId Case ID
     */
    function getFraudCase(bytes32 _caseId) external view returns (FraudCase memory) {
        require(fraudCases[_caseId].timestamp != 0, "Case does not exist");
        return fraudCases[_caseId];
    }
    
    /**
     * @dev Get all fraud cases for a user
     * @param _userIdHash Hashed user ID
     */
    function getUserFraudCases(bytes32 _userIdHash) external view returns (bytes32[] memory) {
        return userFraudCases[_userIdHash];
    }
    
    /**
     * @dev Get case ID by transaction hash
     * @param _transactionIdHash Hashed transaction ID
     */
    function getCaseByTransaction(bytes32 _transactionIdHash) external view returns (bytes32) {
        return transactionToCaseId[_transactionIdHash];
    }
    
    /**
     * @dev Get total number of cases
     */
    function getTotalCases() external view returns (uint256) {
        return totalCases;
    }
    
    /**
     * @dev Get fraud statistics
     */
    function getFraudStatistics() external view returns (
        uint256 total,
        uint256 confirmed,
        uint256 totalAmount
    ) {
        return (totalCases, confirmedFraudCases, totalFraudAmount);
    }
    
    /**
     * @dev Get cases by status
     * @param _status Case status
     */
    function getCasesByStatus(CaseStatus _status) external view returns (bytes32[] memory) {
        uint256 count = 0;
        
        // Count matching cases
        for (uint256 i = 0; i < allCaseIds.length; i++) {
            if (fraudCases[allCaseIds[i]].status == _status) {
                count++;
            }
        }
        
        // Create result array
        bytes32[] memory result = new bytes32[](count);
        uint256 index = 0;
        
        for (uint256 i = 0; i < allCaseIds.length; i++) {
            if (fraudCases[allCaseIds[i]].status == _status) {
                result[index] = allCaseIds[i];
                index++;
            }
        }
        
        return result;
    }
    
    /**
     * @dev Get cases by fraud type
     * @param _fraudType Fraud type
     */
    function getCasesByType(FraudType _fraudType) external view returns (bytes32[] memory) {
        uint256 count = 0;
        
        for (uint256 i = 0; i < allCaseIds.length; i++) {
            if (fraudCases[allCaseIds[i]].fraudType == _fraudType) {
                count++;
            }
        }
        
        bytes32[] memory result = new bytes32[](count);
        uint256 index = 0;
        
        for (uint256 i = 0; i < allCaseIds.length; i++) {
            if (fraudCases[allCaseIds[i]].fraudType == _fraudType) {
                result[index] = allCaseIds[i];
                index++;
            }
        }
        
        return result;
    }
    
    /**
     * @dev Pause contract
     */
    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }
    
    /**
     * @dev Unpause contract
     */
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
    
    /**
     * @dev Grant fraud analyst role
     * @param _analyst Address to grant role
     */
    function grantFraudAnalystRole(address _analyst) external onlyRole(DEFAULT_ADMIN_ROLE) {
        grantRole(FRAUD_ANALYST_ROLE, _analyst);
    }
    
    /**
     * @dev Revoke fraud analyst role
     * @param _analyst Address to revoke role
     */
    function revokeFraudAnalystRole(address _analyst) external onlyRole(DEFAULT_ADMIN_ROLE) {
        revokeRole(FRAUD_ANALYST_ROLE, _analyst);
    }
    
    /**
     * @dev Grant auditor role
     * @param _auditor Address to grant role
     */
    function grantAuditorRole(address _auditor) external onlyRole(DEFAULT_ADMIN_ROLE) {
        grantRole(AUDITOR_ROLE, _auditor);
    }
}
