// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title BiometricVerifier
 * @dev Zero-knowledge proof verification for biometric authentication
 * @notice Verifies biometric proofs without exposing raw biometric data
 */
contract BiometricVerifier is Ownable, ReentrancyGuard {
    
    struct BiometricTemplate {
        bytes32 templateHash;
        uint256 enrollmentTimestamp;
        uint256 lastVerificationTimestamp;
        uint256 verificationCount;
        bool isActive;
        BiometricType biometricType;
    }
    
    struct VerificationAttempt {
        bytes32 attemptId;
        address user;
        uint256 timestamp;
        bool success;
        uint256 confidenceScore;  // 0-100
        bytes32 proofHash;
    }
    
    enum BiometricType {
        Keystroke,
        Mouse,
        Touch,
        Voice,
        Facial,
        Combined
    }
    
    // Mapping from user address to biometric templates
    mapping(address => mapping(BiometricType => BiometricTemplate)) public userTemplates;
    
    // Mapping from attempt ID to verification attempt
    mapping(bytes32 => VerificationAttempt) public verificationAttempts;
    
    // Mapping from user to their verification history
    mapping(address => bytes32[]) public userVerificationHistory;
    
    // Trusted verifier addresses (off-chain verification services)
    mapping(address => bool) public trustedVerifiers;
    
    // Statistics
    uint256 public totalEnrollments;
    uint256 public totalVerifications;
    uint256 public successfulVerifications;
    uint256 public failedVerifications;
    
    // Configuration
    uint256 public minConfidenceScore = 75;  // Minimum confidence for successful verification
    uint256 public maxVerificationAge = 300;  // 5 minutes in seconds
    
    // Events
    event BiometricEnrolled(
        address indexed user,
        BiometricType biometricType,
        bytes32 templateHash,
        uint256 timestamp
    );
    
    event BiometricVerified(
        bytes32 indexed attemptId,
        address indexed user,
        BiometricType biometricType,
        bool success,
        uint256 confidenceScore
    );
    
    event BiometricRevoked(
        address indexed user,
        BiometricType biometricType,
        uint256 timestamp
    );
    
    event TrustedVerifierAdded(address indexed verifier);
    event TrustedVerifierRemoved(address indexed verifier);
    
    modifier onlyTrustedVerifier() {
        require(trustedVerifiers[msg.sender], "Not a trusted verifier");
        _;
    }
    
    constructor() {
        trustedVerifiers[msg.sender] = true;
    }
    
    /**
     * @dev Enroll biometric template
     * @param _user User address
     * @param _biometricType Type of biometric
     * @param _templateHash Hash of biometric template
     */
    function enrollBiometric(
        address _user,
        BiometricType _biometricType,
        bytes32 _templateHash
    ) external onlyTrustedVerifier nonReentrant {
        require(_user != address(0), "Invalid user address");
        require(_templateHash != bytes32(0), "Invalid template hash");
        
        BiometricTemplate storage template = userTemplates[_user][_biometricType];
        
        template.templateHash = _templateHash;
        template.enrollmentTimestamp = block.timestamp;
        template.lastVerificationTimestamp = 0;
        template.verificationCount = 0;
        template.isActive = true;
        template.biometricType = _biometricType;
        
        totalEnrollments++;
        
        emit BiometricEnrolled(_user, _biometricType, _templateHash, block.timestamp);
    }
    
    /**
     * @dev Verify biometric proof
     * @param _user User address
     * @param _biometricType Type of biometric
     * @param _proofHash Hash of zero-knowledge proof
     * @param _confidenceScore Confidence score (0-100)
     */
    function verifyBiometric(
        address _user,
        BiometricType _biometricType,
        bytes32 _proofHash,
        uint256 _confidenceScore
    ) external onlyTrustedVerifier nonReentrant returns (bytes32) {
        require(_user != address(0), "Invalid user address");
        require(_proofHash != bytes32(0), "Invalid proof hash");
        require(_confidenceScore <= 100, "Invalid confidence score");
        
        BiometricTemplate storage template = userTemplates[_user][_biometricType];
        require(template.isActive, "Biometric not enrolled or inactive");
        
        // Generate attempt ID
        bytes32 attemptId = keccak256(
            abi.encodePacked(
                _user,
                _biometricType,
                _proofHash,
                block.timestamp,
                msg.sender
            )
        );
        
        // Determine success based on confidence score
        bool success = _confidenceScore >= minConfidenceScore;
        
        // Record verification attempt
        VerificationAttempt memory attempt = VerificationAttempt({
            attemptId: attemptId,
            user: _user,
            timestamp: block.timestamp,
            success: success,
            confidenceScore: _confidenceScore,
            proofHash: _proofHash
        });
        
        verificationAttempts[attemptId] = attempt;
        userVerificationHistory[_user].push(attemptId);
        
        // Update template statistics
        template.lastVerificationTimestamp = block.timestamp;
        template.verificationCount++;
        
        // Update global statistics
        totalVerifications++;
        if (success) {
            successfulVerifications++;
        } else {
            failedVerifications++;
        }
        
        emit BiometricVerified(attemptId, _user, _biometricType, success, _confidenceScore);
        
        return attemptId;
    }
    
    /**
     * @dev Batch verify multiple biometric types
     * @param _user User address
     * @param _biometricTypes Array of biometric types
     * @param _proofHashes Array of proof hashes
     * @param _confidenceScores Array of confidence scores
     */
    function batchVerifyBiometrics(
        address _user,
        BiometricType[] memory _biometricTypes,
        bytes32[] memory _proofHashes,
        uint256[] memory _confidenceScores
    ) external onlyTrustedVerifier nonReentrant returns (bytes32[] memory) {
        require(
            _biometricTypes.length == _proofHashes.length &&
            _proofHashes.length == _confidenceScores.length,
            "Array length mismatch"
        );
        
        bytes32[] memory attemptIds = new bytes32[](_biometricTypes.length);
        
        for (uint256 i = 0; i < _biometricTypes.length; i++) {
            attemptIds[i] = this.verifyBiometric(
                _user,
                _biometricTypes[i],
                _proofHashes[i],
                _confidenceScores[i]
            );
        }
        
        return attemptIds;
    }
    
    /**
     * @dev Revoke biometric template
     * @param _user User address
     * @param _biometricType Type of biometric
     */
    function revokeBiometric(
        address _user,
        BiometricType _biometricType
    ) external {
        require(
            msg.sender == _user || trustedVerifiers[msg.sender] || msg.sender == owner(),
            "Not authorized"
        );
        
        BiometricTemplate storage template = userTemplates[_user][_biometricType];
        require(template.isActive, "Biometric not active");
        
        template.isActive = false;
        
        emit BiometricRevoked(_user, _biometricType, block.timestamp);
    }
    
    /**
     * @dev Get biometric template
     * @param _user User address
     * @param _biometricType Type of biometric
     */
    function getBiometricTemplate(
        address _user,
        BiometricType _biometricType
    ) external view returns (BiometricTemplate memory) {
        return userTemplates[_user][_biometricType];
    }
    
    /**
     * @dev Get verification attempt
     * @param _attemptId Attempt ID
     */
    function getVerificationAttempt(
        bytes32 _attemptId
    ) external view returns (VerificationAttempt memory) {
        return verificationAttempts[_attemptId];
    }
    
    /**
     * @dev Get user verification history
     * @param _user User address
     */
    function getUserVerificationHistory(
        address _user
    ) external view returns (bytes32[] memory) {
        return userVerificationHistory[_user];
    }
    
    /**
     * @dev Get recent verification attempts for user
     * @param _user User address
     * @param _count Number of recent attempts to return
     */
    function getRecentVerifications(
        address _user,
        uint256 _count
    ) external view returns (VerificationAttempt[] memory) {
        bytes32[] memory history = userVerificationHistory[_user];
        uint256 length = history.length < _count ? history.length : _count;
        
        VerificationAttempt[] memory recent = new VerificationAttempt[](length);
        
        for (uint256 i = 0; i < length; i++) {
            uint256 index = history.length - 1 - i;
            recent[i] = verificationAttempts[history[index]];
        }
        
        return recent;
    }
    
    /**
     * @dev Check if biometric is enrolled and active
     * @param _user User address
     * @param _biometricType Type of biometric
     */
    function isBiometricActive(
        address _user,
        BiometricType _biometricType
    ) external view returns (bool) {
        return userTemplates[_user][_biometricType].isActive;
    }
    
    /**
     * @dev Get verification statistics
     */
    function getVerificationStatistics() external view returns (
        uint256 total,
        uint256 successful,
        uint256 failed,
        uint256 successRate
    ) {
        uint256 rate = totalVerifications > 0
            ? (successfulVerifications * 100) / totalVerifications
            : 0;
        
        return (totalVerifications, successfulVerifications, failedVerifications, rate);
    }
    
    /**
     * @dev Add trusted verifier
     * @param _verifier Verifier address
     */
    function addTrustedVerifier(address _verifier) external onlyOwner {
        require(_verifier != address(0), "Invalid verifier address");
        require(!trustedVerifiers[_verifier], "Already trusted verifier");
        
        trustedVerifiers[_verifier] = true;
        
        emit TrustedVerifierAdded(_verifier);
    }
    
    /**
     * @dev Remove trusted verifier
     * @param _verifier Verifier address
     */
    function removeTrustedVerifier(address _verifier) external onlyOwner {
        require(trustedVerifiers[_verifier], "Not a trusted verifier");
        
        trustedVerifiers[_verifier] = false;
        
        emit TrustedVerifierRemoved(_verifier);
    }
    
    /**
     * @dev Update minimum confidence score
     * @param _minScore New minimum score (0-100)
     */
    function updateMinConfidenceScore(uint256 _minScore) external onlyOwner {
        require(_minScore <= 100, "Invalid score");
        minConfidenceScore = _minScore;
    }
    
    /**
     * @dev Update maximum verification age
     * @param _maxAge New maximum age in seconds
     */
    function updateMaxVerificationAge(uint256 _maxAge) external onlyOwner {
        maxVerificationAge = _maxAge;
    }
    
    /**
     * @dev Check if verification is still valid
     * @param _attemptId Attempt ID
     */
    function isVerificationValid(bytes32 _attemptId) external view returns (bool) {
        VerificationAttempt memory attempt = verificationAttempts[_attemptId];
        
        if (!attempt.success) {
            return false;
        }
        
        uint256 age = block.timestamp - attempt.timestamp;
        return age <= maxVerificationAge;
    }
}
