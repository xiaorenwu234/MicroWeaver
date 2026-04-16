import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.AnnotationDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.ConditionalExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.FieldAccessExpr;
import com.github.javaparser.ast.expr.LambdaExpr;
import com.github.javaparser.ast.expr.MemberValuePair;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.MethodReferenceExpr;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.WildcardType;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.type.PrimitiveType;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.type.VoidType;
import com.github.javaparser.Position;
import com.github.javaparser.resolution.Context;
import com.github.javaparser.resolution.UnsolvedSymbolException;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedAnnotationDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedFieldDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedReferenceTypeDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedTypeDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedValueDeclaration;
import com.github.javaparser.resolution.types.ResolvedArrayType;
import com.github.javaparser.resolution.types.ResolvedReferenceType;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.resolution.types.ResolvedWildcard;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.javaparsermodel.JavaParserFacade;
import com.github.javaparser.symbolsolver.javaparsermodel.JavaParserFactory;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.resolution.model.SymbolReference;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Class-level dependency analyzer
 * Analyzes dependencies between classes, including:
 * - Field type dependencies
 * - Method parameter type dependencies
 * - Method return type dependencies
 * - Inheritance relationships
 * - Interface implementation relationships
 * - Types used inside methods
 */
public class ClassDependencyAnalyzer {
    
    /**
     * Class information data structure, unified storage of all class information
     */
    public static class ClassInfo {
        private String qualifiedName;           // Fully qualified class name
        private String filePath;                // Relative file path
        private String typeKind;                // Type: class, interface, enum, annotation
        private List<String> dependencies;      // Dependency list
        private List<String> extendsAndImplements; // Inheritance and implementation relationships
        private String javaDoc;                 // JavaDoc information
        private List<String> methods;          // Method signature list
        
        public ClassInfo() {
            this.dependencies = new ArrayList<>();
            this.extendsAndImplements = new ArrayList<>();
            this.methods = new ArrayList<>();
            this.javaDoc = "";
        }
        
        // Getters and Setters
        public String getQualifiedName() { return qualifiedName; }
        public void setQualifiedName(String qualifiedName) { this.qualifiedName = qualifiedName; }
        
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        
        public String getTypeKind() { return typeKind; }
        public void setTypeKind(String typeKind) { this.typeKind = typeKind; }
        
        public List<String> getDependencies() { return dependencies; }
        public void setDependencies(List<String> dependencies) { this.dependencies = dependencies; }
        
        public List<String> getExtendsAndImplements() { return extendsAndImplements; }
        public void setExtendsAndImplements(List<String> extendsAndImplements) { this.extendsAndImplements = extendsAndImplements; }
        
        public String getJavaDoc() { return javaDoc; }
        public void setJavaDoc(String javaDoc) { this.javaDoc = javaDoc; }
        
        public List<String> getMethods() { return methods; }
        public void setMethods(List<String> methods) { this.methods = methods; }
    }
    
    // Project package prefix, used to filter and keep only in-project class dependencies
    private static String projectPackagePrefix = null;

    private static Boolean collectAllDependencies = false;
    
    // Error message collector (thread-safe)
    private static final ConcurrentLinkedQueue<String> errorMessages = new ConcurrentLinkedQueue<>();
    
    // TypeSolver references for determining type source
    private static ReflectionTypeSolver reflectionTypeSolver = null;
    private static List<JavaParserTypeSolver> javaParserTypeSolvers = new ArrayList<>();
    private static List<JarTypeSolver> jarTypeSolvers = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        String targetProjectRoot = args[0];
        System.out.println("Target project root: " + targetProjectRoot);

        String classInfoPath = args[1];
        System.out.println("Output path: " + classInfoPath);

        if (args.length > 2) {
            projectPackagePrefix = args[2];
            System.out.println("Project package prefix: " + projectPackagePrefix);
        }

        // Recursively find all src/main/java directories (supports multi-module projects)
        File projectRootDir = new File(targetProjectRoot);
        List<File> sourceDirs = findAllSourceDirs(projectRootDir);
        
        if (sourceDirs.isEmpty()) {
            System.err.println("Error: No src/main/java directory found in project root");
            System.exit(1);
        }
        
        System.out.println("Found " + sourceDirs.size() + " source directories:");
        for (File dir : sourceDirs) {
            System.out.println("  - " + dir.getAbsolutePath());
        }

        // Create CombinedTypeSolver
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        
        // Save ReflectionTypeSolver reference
        reflectionTypeSolver = new ReflectionTypeSolver();
        combinedTypeSolver.add(reflectionTypeSolver);
        
        // Save JavaParserTypeSolver reference
        JavaParserTypeSolver rootJavaParserTypeSolver = new JavaParserTypeSolver(projectRootDir);
        javaParserTypeSolvers.add(rootJavaParserTypeSolver);
        combinedTypeSolver.add(rootJavaParserTypeSolver);
        
        // Add JavaParserTypeSolver for each source directory
        for (File sourceDir : sourceDirs) {
            JavaParserTypeSolver sourceTypeSolver = new JavaParserTypeSolver(sourceDir);
            javaParserTypeSolvers.add(sourceTypeSolver);
            combinedTypeSolver.add(sourceTypeSolver);
        }

        // deleteDependencyDirsRecursively(projectRootDir);

        loadStaticDependencies(combinedTypeSolver, projectRootDir);
        System.out.println("Static dependencies loaded");

        // Collect Java files from all source directories
        List<File> javaFiles = new ArrayList<>();
        for (File sourceDir : sourceDirs) {
            javaFiles.addAll(listJavaFilesRecursively(sourceDir));
        }
        System.out.println("Found " + javaFiles.size() + " Java files");

        // if (javaFiles.size() > 200) {
        //     javaFiles = javaFiles.subList(0, 200);
        // }

        // Unified class info storage: class name -> ClassInfo (using thread-safe collection)
        Map<String, ClassInfo> classInfoMap = new ConcurrentHashMap<>();

        // Get available processor count for parallel processing
        int numThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("Using " + numThreads + " threads for parallel processing");
        
        // Create thread pool
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger processedCount = new AtomicInteger(0);
        
        // Extract file count as final variable for use in lambda
        final int totalFiles = javaFiles.size();
        
        // Submit tasks for each file
        for (File javaFile : javaFiles) {
            executor.submit(() -> {
                try {
                    // Create independent JavaParser and JavaParserFacade instances for each thread
                    // Because JavaParser may not be thread-safe
                    JavaParser parser = new JavaParser();
                    parser.getParserConfiguration().setSymbolResolver(new JavaSymbolSolver(combinedTypeSolver));
                    JavaParserFacade facade = JavaParserFacade.get(combinedTypeSolver);
                    
                    int current = processedCount.incrementAndGet();
                    if (current % 100 == 0) {
                        System.out.println("Processed " + current + "/" + totalFiles + " files");
                    }
                    
                    analyzeDependencies(parser, facade, javaFile, classInfoMap, projectRootDir, projectPackagePrefix);
                } catch (Exception e) {
                    logError(javaFile, "File analysis failed", null, e, true);
                }
            });
        }
        
        // Shutdown thread pool and wait for all tasks to complete
        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                executor.shutdownNow();
                System.err.println("Warning: processing timeout, force shutting down thread pool");
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        System.out.println("All files processing completed");

        createParentDirectory(classInfoPath);
        writeClassInfoToFile(classInfoMap, classInfoPath);

        // Output statistics
        System.out.println("\n=== Statistics ===");
        System.out.println("Total classes: " + classInfoMap.size());
        long withDependencies = classInfoMap.values().stream().filter(info -> !info.getDependencies().isEmpty()).count();
        long withExtendsAndImplements = classInfoMap.values().stream().filter(info -> !info.getExtendsAndImplements().isEmpty()).count();
        long withMethods = classInfoMap.values().stream().filter(info -> !info.getMethods().isEmpty()).count();
        System.out.println("Classes with dependencies: " + withDependencies);
        System.out.println("Classes with inheritance/implementation: " + withExtendsAndImplements);
        System.out.println("Classes with method info: " + withMethods);
        System.out.println("Total errors: " + errorMessages.size());
    }


    public static void createParentDirectory(String filePath) {
        // 1. Convert file path to Path object (Java NIO recommended approach, easier to use)
        Path filePathObj = Paths.get(filePath);

        // 2. Get parent directory path (e.g. D:/data/json)
        Path parentDir = filePathObj.getParent();

        // 3. Check if parent path is null (avoid case where filePath is root/has no parent)
        if (parentDir == null) {
            System.out.println("File path has no parent directory, no need to create");
            return;
        }

        // 4. Check if directory exists, create if not (mkdirs() creates multi-level directories)
        if (!Files.exists(parentDir)) {
            try {
                // mkdirs(): Create multi-level directories (e.g. when data doesn't exist, creates both data and json)
                // Unlike mkdir(): Only creates a single-level directory, fails if parent doesn't exist
                Files.createDirectories(parentDir);
                System.out.println("Directory created successfully: " + parentDir);
            } catch (Exception e) {
                // Catch exception (e.g. insufficient permissions, invalid path, etc.)
                System.err.println("Failed to create directory: " + parentDir);
                e.printStackTrace();
            }
        } else {
            System.out.println("Directory already exists: " + parentDir);
        }
    }

    /**
     * Analyze class dependencies for a single file
     */
    private static void analyzeDependencies(JavaParser parser, JavaParserFacade facade,
                                                  File javaFile, Map<String, ClassInfo> classInfoMap,
                                                  File projectRootDir, String projectPackagePrefix) {
        // System.out.println("Analyzing file: " + javaFile.getAbsolutePath());
        try (FileInputStream in = new FileInputStream(javaFile)) {
            CompilationUnit cu = parser.parse(in).getResult().orElse(null);
            if (cu == null) {
                return;
            }

            // Get current file's package name
            String packageName = cu.getPackageDeclaration()
                    .map(p -> p.getNameAsString())
                    .orElse("");
            
            // Early check: if package prefix is set and package name doesn't match, skip entire file
            // Note: default package (empty packageName) cannot be checked early, needs to check class name later
            if (projectPackagePrefix != null && !projectPackagePrefix.isEmpty() && !packageName.isEmpty()) {
                // Package name should start with projectPackagePrefix (plus ".") or equal projectPackagePrefix
                if (!packageName.startsWith(projectPackagePrefix + ".") && !packageName.equals(projectPackagePrefix)) {
                    return; // Package name doesn't match prefix, skip entire file
                }
            }
            
            // Get all top-level type declarations (class, interface, enum, annotation, Record) - excluding inner classes
            List<TypeDeclaration<?>> topLevelTypes = cu.getTypes();
            
            // Calculate file relative path
            String fileRelativePath = getRelativePath(javaFile, projectRootDir);
            
            // Iterate over all top-level type declarations (class, interface, enum, annotation, Record)
            // All TypeDeclaration subclasses can be handled uniformly, no need to differentiate
            for (TypeDeclaration<?> typeDecl : topLevelTypes) {
                try {
                    String typeName = getFullName(typeDecl, packageName);
                    
                    // Get or create ClassInfo
                    ClassInfo classInfo = classInfoMap.computeIfAbsent(typeName, k -> {
                        ClassInfo info = new ClassInfo();
                        info.setQualifiedName(typeName);
                        info.setFilePath(fileRelativePath);
                        info.setTypeKind(getTypeKind(typeDecl));
                        return info;
                    });
                    
                    // Set file path and type (if previously existed but info was incomplete)
                    if (classInfo.getFilePath() == null || classInfo.getFilePath().isEmpty()) {
                        classInfo.setFilePath(fileRelativePath);
                    }
                    if (classInfo.getTypeKind() == null || classInfo.getTypeKind().isEmpty()) {
                        classInfo.setTypeKind(getTypeKind(typeDecl));
                    }
                    
                    // Extract JavaDoc
                    if (typeDecl.getJavadoc().isPresent()) {
                        String javadocText = typeDecl.getJavadoc().get().toText();
                        if (!javadocText.isEmpty()) {
                            classInfo.setJavaDoc(javadocText);
                        }
                    }
                    
                    // Use thread-safe collection to store dependencies
                    Set<String> dependencies = ConcurrentHashMap.newKeySet();
                    Set<String> extendsAndImplements = ConcurrentHashMap.newKeySet();
                    
                    // Extract inheritance and implementation relationships specifically (including inner classes)
                    analyzeExtendsAndImplements(typeDecl, facade, extendsAndImplements, javaFile);
                    
                    // Extract top-level type method signatures (excluding inner class methods)
                    analyzeMethods(typeDecl, facade, classInfo.getMethods(), javaFile);
                    
                    // Unified handling for all TypeDeclaration: get all Types and Expressions
                    for (ClassOrInterfaceType type : typeDecl.findAll(ClassOrInterfaceType.class)) {
                        analyzeType(type, facade, dependencies, javaFile);
                    }
                    for (Expression expression : typeDecl.findAll(Expression.class)) {
                        analyzeExpression(expression, facade, dependencies, javaFile);
                    }
                    
                    // Convert Set to List and filter project dependencies
                    classInfo.setDependencies(new ArrayList<>(filterProjectDependencies(dependencies, typeName)));
                    classInfo.setExtendsAndImplements(new ArrayList<>(filterProjectDependencies(extendsAndImplements, typeName)));

                } catch (Exception e) {
                    String typeName = getFullName(typeDecl, packageName);
                    logError(javaFile, "Type analysis failed", typeName, e, false);
                }
            }

        } catch (Exception e) {
            logError(javaFile, "File read failed", null, e, true);
        }
    }

    private static void analyzeExtendsAndImplements(TypeDeclaration<?> typeDecl, JavaParserFacade facade, Set<String> extendsAndImplements, File javaFile) {
        // System.out.println("Analyzing inheritance and implementation: " + typeDecl.getNameAsString());
        
        try {
            for (ClassOrInterfaceDeclaration decl : typeDecl.findAll(ClassOrInterfaceDeclaration.class)) {
                // System.out.println("Found class definition: " + decl.getNameAsString());
                for (ClassOrInterfaceType type : decl.getExtendedTypes()) {
                    // System.out.println("Found extended type: " + type.getNameAsString());
                    analyzeType(type, facade, extendsAndImplements, javaFile);
                }
                for (ClassOrInterfaceType type : decl.getImplementedTypes()) {
                    // System.out.println("Found implemented type: " + type.getNameAsString());
                    analyzeType(type, facade, extendsAndImplements, javaFile);
                }
            }
        } catch (Exception e) {
            logError(javaFile, "Inheritance and implementation analysis failed", typeDecl.getNameAsString(), e, false);
        }
    }

    /**
     * Analyze methods of top-level type, extract complete method signatures
     * Only extracts methods directly defined by the top-level type, excluding inner class methods
     */
    private static void analyzeMethods(TypeDeclaration<?> typeDecl, JavaParserFacade facade, List<String> methods, File javaFile) {
        try {
            // Only get methods directly defined by the top-level type, excluding inner class methods
            // Use getMethods() instead of findAll(), because findAll would recursively find all methods including inner classes
            List<MethodDeclaration> methodDecls = new ArrayList<>();
            
            if (typeDecl instanceof ClassOrInterfaceDeclaration) {
                ClassOrInterfaceDeclaration decl = (ClassOrInterfaceDeclaration) typeDecl;
                methodDecls.addAll(decl.getMethods());
            } else if (typeDecl instanceof EnumDeclaration) {
                EnumDeclaration enumDecl = (EnumDeclaration) typeDecl;
                methodDecls.addAll(enumDecl.getMethods());
            } else if (typeDecl instanceof AnnotationDeclaration) {
                AnnotationDeclaration annotationDecl = (AnnotationDeclaration) typeDecl;
                methodDecls.addAll(annotationDecl.getMethods());
            }
            
            // Process all found methods
            for (MethodDeclaration methodDecl : methodDecls) {
                try {
                    String methodSignature = methodDecl.getDeclarationAsString();
                    if (methodSignature != null && !methodSignature.isEmpty()) {
                        methods.add(methodSignature);
                    }
                } catch (Exception e) {
                    logError(javaFile, "Method signature analysis failed", methodDecl.getNameAsString(), e, false);
                }
            }
        } catch (Exception e) {
            logError(javaFile, "Method analysis failed", typeDecl.getNameAsString(), e, false);
        }
    }

    /**
     * Build complete method signature
     * Format: ReturnType methodName(ParamType1, ParamType2, ...)
     */
    private static String buildMethodSignature(MethodDeclaration methodDecl, JavaParserFacade facade) {
        try {
            StringBuilder signature = new StringBuilder();
            
            // Get return type
            Type returnType = methodDecl.getType();
            String returnTypeStr = getTypeString(returnType, facade);
            signature.append(returnTypeStr).append(" ");
            
            // Get method name
            signature.append(methodDecl.getNameAsString());
            
            // Get parameter list
            signature.append("(");
            NodeList<Parameter> parameters = methodDecl.getParameters();
            for (int i = 0; i < parameters.size(); i++) {
                if (i > 0) {
                    signature.append(", ");
                }
                Parameter param = parameters.get(i);
                Type paramType = param.getType();
                String paramTypeStr = getTypeString(paramType, facade);
                signature.append(paramTypeStr);
            }
            signature.append(")");
            
            return signature.toString();
        } catch (Exception e) {
            // If parsing fails, try building signature in a simple way
            try {
                StringBuilder signature = new StringBuilder();
                signature.append(methodDecl.getType().toString()).append(" ");
                signature.append(methodDecl.getNameAsString()).append("(");
                NodeList<Parameter> parameters = methodDecl.getParameters();
                for (int i = 0; i < parameters.size(); i++) {
                    if (i > 0) {
                        signature.append(", ");
                    }
                    signature.append(parameters.get(i).getType().toString());
                }
                signature.append(")");
                return signature.toString();
            } catch (Exception e2) {
                return null;
            }
        }
    }

    /**
     * Get string representation of a type, attempting to resolve to fully qualified name
     */
    private static String getTypeString(Type type, JavaParserFacade facade) {
        try {
            // Try to resolve type to get fully qualified name
            ResolvedType resolvedType = facade.convertToUsage(type);
            if (resolvedType.isReferenceType()) {
                return resolvedType.asReferenceType().getQualifiedName();
            } else if (resolvedType.isPrimitive()) {
                return resolvedType.asPrimitive().name().toLowerCase();
            } else if (resolvedType.isVoid()) {
                return "void";
            } else if (resolvedType.isArray()) {
                ResolvedArrayType arrayType = resolvedType.asArrayType();
                ResolvedType componentType = arrayType.getComponentType();
                String componentTypeStr = getTypeStringFromResolved(componentType);
                return componentTypeStr + "[]";
            } else {
                return type.toString();
            }
        } catch (Exception e) {
            // Resolution failed, return original type string
            return type.toString();
        }
    }

    /**
     * Get type string from ResolvedType
     */
    private static String getTypeStringFromResolved(ResolvedType resolvedType) {
        try {
            if (resolvedType.isReferenceType()) {
                return resolvedType.asReferenceType().getQualifiedName();
            } else if (resolvedType.isPrimitive()) {
                return resolvedType.asPrimitive().name().toLowerCase();
            } else if (resolvedType.isVoid()) {
                return "void";
            } else {
                return resolvedType.toString();
            }
        } catch (Exception e) {
            return resolvedType.toString();
        }
    }

    private static void analyzeType(ClassOrInterfaceType type, JavaParserFacade facade, Set<String> dependencies, File javaFile) {
        try {
            // System.out.println("Analyzing type: " + type.toString());
            if (type.getParentNode().isPresent() && type.getParentNode().get() instanceof ClassOrInterfaceType) {
                // This is the middle part of a qualified name (package name), skip, do not resolve
                return;
            }

            // Use facade to resolve type instead of calling type.resolve() directly
            // Because facade is associated with TypeSolver and can correctly resolve types
            ResolvedType resolvedType = facade.convertToUsage(type);
            // ResolvedType resolvedType = type.resolve();

            analyzeResolvedType(resolvedType, dependencies);
        } catch (Exception e) {
            // System.out.println("Type analysis failed: " + type.toString() + " " + e.getMessage());
            if (type.getTypeArguments().isPresent()) {
                NodeList<Type> typeArgs = type.getTypeArguments().get();
            
                if (!typeArgs.isEmpty()) {
                    type.setTypeArguments(new NodeList<>());
                    
                    try {
                        analyzeType(type, facade, dependencies, javaFile);
                        return;
                    } catch (Exception e2) {
                        e = e2;
                    }
                }
            }

            // Inner class
            if (type.getScope().isPresent()) {
                try {
                    solveInnerClassChain(type, facade, dependencies);
                    return;
                } catch (Exception e1) {
                    // Cannot resolve, skip
                }
            }

            // Check if the error is caused by package name or variable name (should be handled silently)
            String typeName = type.toString();
            
            // Other type resolution errors, log error (may be a real configuration issue)
            logError(javaFile, "Type analysis failed", typeName, e, false);
        }
    }

    private static String solveInnerClassChain(ClassOrInterfaceType type, JavaParserFacade facade, Set<String> dependencies) {
        Optional<ClassOrInterfaceType> scope = type.getScope();
        if (scope.isPresent()) {
            String outerName = "";
            try {
                // Use facade method to resolve type
                ResolvedType resolvedType = facade.convertToUsage(scope.get());
                ResolvedReferenceType refType = resolvedType.asReferenceType();
                outerName = refType.getQualifiedName();
                if (collectAllDependencies || determineTypeSolverSource(outerName).equals("JavaParserTypeSolver")) {
                    dependencies.add(outerName);
                } else {
                    return null;
                }
            } catch (Exception e) {
                outerName = solveInnerClassChain(scope.get(), facade, dependencies);
                if (outerName == null) {
                    return null;
                }
            }
            String thisQualifiedName = outerName + "." + type.getNameAsString();
            dependencies.add(thisQualifiedName);
            return  thisQualifiedName;
        } else {
            throw new RuntimeException("Cannot resolve inner class chain");
        }
    }
    
    private static void analyzeResolvedType(ResolvedType resolvedType, Set<String> dependencies) {
        try {
            ResolvedReferenceType refType = resolvedType.asReferenceType();
            String typeName = refType.getQualifiedName();
            if (collectAllDependencies || determineTypeSolverSource(typeName).equals("JavaParserTypeSolver")) {
                dependencies.add(typeName);
            }
            
            // Optional: determine type source (uncomment below if needed)
            // String typeSolverSource = determineTypeSolverSource(typeName);
            // System.out.println("Type " + typeName + " resolved by " + typeSolverSource);
        } catch (Exception e) {
            // Cannot resolve, skip
            throw e;
        }
    }
    
    /**
     * Determine which TypeSolver resolved the type
     * @param qualifiedName Fully qualified name of the type
     * @return TypeSolver type string: "ReflectionTypeSolver", "JavaParserTypeSolver", "JarTypeSolver" or "Unknown"
     */
    public static String determineTypeSolverSource(String qualifiedName) {
        if (qualifiedName == null || qualifiedName.isEmpty()) {
            return "Unknown";
        }
        
        // For java.* and javax.* packages: Java spec reserves these package names, project code cannot define classes in them
        // So use ReflectionTypeSolver directly (fastest and only possibility)
        if (qualifiedName.startsWith("java.") || qualifiedName.startsWith("javax.")) {
            if (reflectionTypeSolver != null) {
                try {
                    SymbolReference<ResolvedReferenceTypeDeclaration> symbolRef = 
                        reflectionTypeSolver.tryToSolveType(qualifiedName);
                    if (symbolRef.isSolved()) {
                        return "ReflectionTypeSolver";
                    }
                } catch (Exception e) {
                    // Resolution failed, return Unknown
                }
            }
            // If ReflectionTypeSolver cannot resolve java.* or javax.* class, return Unknown
            return "Unknown";
        }
        
        // For other classes: try each TypeSolver in order
        // 1. First try JavaParserTypeSolver (project classes) - this is what we care about most
        for (JavaParserTypeSolver javaParserTypeSolver : javaParserTypeSolvers) {
            try {
                SymbolReference<ResolvedReferenceTypeDeclaration> symbolRef = 
                    javaParserTypeSolver.tryToSolveType(qualifiedName);
                if (symbolRef.isSolved()) {
                    return "JavaParserTypeSolver";
                }
            } catch (Exception e) {
                // Resolution failed, continue trying next
            }
        }
        
        // 2. Then try ReflectionTypeSolver (unlikely, but for completeness, e.g. some non-standard JDK classes)
        if (reflectionTypeSolver != null) {
            try {
                SymbolReference<ResolvedReferenceTypeDeclaration> symbolRef = 
                    reflectionTypeSolver.tryToSolveType(qualifiedName);
                if (symbolRef.isSolved()) {
                    return "ReflectionTypeSolver";
                }
            } catch (Exception e) {
                // Resolution failed, continue trying next
            }
        }
        
        // 3. Finally try JarTypeSolver (third-party dependencies)
        for (JarTypeSolver jarTypeSolver : jarTypeSolvers) {
            try {
                SymbolReference<ResolvedReferenceTypeDeclaration> symbolRef = 
                    jarTypeSolver.tryToSolveType(qualifiedName);
                if (symbolRef.isSolved()) {
                    return "JarTypeSolver";
                }
            } catch (Exception e) {
                // Resolution failed, continue trying next
            }
        }
        
        // If all TypeSolvers cannot resolve, return Unknown
        return "Unknown";
    }

    private static void analyzeResolvedTypeDeclaration(ResolvedTypeDeclaration resolvedTypeDeclaration, Set<String> dependencies) {
        try {
            String typeName = resolvedTypeDeclaration.getQualifiedName();
            // Extract type itself (e.g. ClassA)
            if (collectAllDependencies || determineTypeSolverSource(typeName).equals("JavaParserTypeSolver")) {
                dependencies.add(typeName);
            }
        } catch (Exception e) {
            // Cannot resolve, skip
            throw e;
        }
    }

    private static void analyzeExpression(Expression expr, JavaParserFacade facade, Set<String> dependencies, File javaFile) {
        try {
            // 1. Annotation expression - need to record annotation type
            if (expr instanceof AnnotationExpr annotationExpr) {
                try {
                    ResolvedAnnotationDeclaration resolved = annotationExpr.resolve();
                    String fqName = resolved.getQualifiedName(); // Fully qualified name of annotation class
                    if (collectAllDependencies || determineTypeSolverSource(fqName).equals("JavaParserTypeSolver")) {
                        dependencies.add(fqName);
                    }
                } catch (Exception e) {
                    // Annotation resolution failed, log error (this is a real error, not a type inference issue)
                    logError(javaFile, "Annotation analysis failed", expr.toString(), e, false);
                }
                return;
            }
            
            // 2. Method call expression - need to analyze method return type and declaring type
            // This info cannot be obtained via findAll(Type.class), as return type may not be in type declarations
            // scope and arguments will be found by findAll(Expression.class), no need for recursive processing
            if (expr instanceof MethodCallExpr methodCall) {
                try {
                    ResolvedMethodDeclaration methodDecl = methodCall.resolve();
                    // Analyze method return type (may not be in type declarations)
                    analyzeResolvedType(methodDecl.getReturnType(), dependencies);
                    // Analyze declaring type of the method
                    analyzeResolvedTypeDeclaration(methodDecl.declaringType(), dependencies);
                    // resolve() succeeded, return directly
                    return;
                } catch (Exception e) {
                    // resolve() failed, don't handle here, fall through to default getType() handling
                }
            }
            
            // 3. Field access expression - need to analyze field type and declaring type
            // This info cannot be obtained via findAll(Type.class)
            // scope will be found by findAll(Expression.class), no need for recursive processing
            if (expr instanceof FieldAccessExpr fieldAccess) {
                try {
                    // resolve() returns ResolvedValueDeclaration, need to convert to ResolvedFieldDeclaration
                    ResolvedValueDeclaration valueDecl = fieldAccess.resolve();
                    if (valueDecl instanceof ResolvedFieldDeclaration fieldDecl) {
                        analyzeResolvedType(fieldDecl.getType(), dependencies);
                        // Analyze declaring type of the field
                        analyzeResolvedTypeDeclaration(fieldDecl.declaringType(), dependencies);
                        // resolve() succeeded, return directly
                        return;
                    }
                } catch (Exception e) {
                    // resolve() failed, don't handle here, fall through to default getType() handling
                }
            }
            
            // 4. Method reference expression - need to analyze the referenced method's declaring type and functional interface type
            // This info cannot be obtained via findAll(Type.class)
            // scope will be found by findAll(Expression.class), no need for recursive processing
            if (expr instanceof MethodReferenceExpr methodRef) {
                // Analyze method declaration (get declaring type of the method)
                try {
                    ResolvedMethodDeclaration methodDecl = methodRef.resolve();
                    // Analyze declaring type of the method
                    analyzeResolvedTypeDeclaration(methodDecl.declaringType(), dependencies);
                    return;
                } catch (Exception e) {
                    // resolve() failed, don't handle here, fall through to default getType() handling
                }
            }
            
            // For other expressions, uniformly use getType() to attempt type resolution
            // Including:
            // - Method call expressions (when resolve() fails)
            // - Field access expressions (when resolve() fails)
            // - Lambda expressions (functional interface type)
            // - Object creation expressions
            // - Ternary operators, binary expressions, etc. (may fail, but handled uniformly)
            // Sub-expressions/sub-types of these expressions are already handled by findAll, here we only try to resolve the expression itself
            // If resolution fails, silently skip (as these expressions may not contain type dependency info)
            try {
                ResolvedType resolvedType = facade.getType(expr);
                if (resolvedType != null) {
                    analyzeResolvedType(resolvedType, dependencies);
                }
            } catch (Exception typeResolveException) {
                // For expressions whose type cannot be resolved, silently skip
            }
        } catch (Exception e) {
            // Check if the error is related to type resolution (should be handled silently)
            String exprStr = expr.toString();
            
            // Other exceptions (e.g. NPE, config issues), log error
            logError(javaFile, "Expression analysis failed", exprStr, e, false);
        }
    }

    /**
     * Get the full name of a type declaration (class, interface, enum, annotation)
     */
    private static String getFullName(TypeDeclaration<?> n, String packageName) {
        String typeName = n.getNameAsString();
        if (!packageName.isEmpty()) {
            return packageName + "." + typeName;
        }
        return typeName;
    }
    
    /**
     * Get type kind (class, interface, enum, annotation)
     */
    private static String getTypeKind(TypeDeclaration<?> typeDecl) {
        if (typeDecl instanceof ClassOrInterfaceDeclaration) {
            ClassOrInterfaceDeclaration decl = (ClassOrInterfaceDeclaration) typeDecl;
            return decl.isInterface() ? "interface" : "class";
        } else if (typeDecl instanceof EnumDeclaration) {
            return "enum";
        } else if (typeDecl instanceof AnnotationDeclaration) {
            return "annotation";
        } else {
            // May be Record or other type
            return "class";
        }
    }
    
    /**
     * Get file path relative to project root directory
     */
    private static String getRelativePath(File file, File projectRootDir) {
        try {
            Path filePath = file.toPath().toAbsolutePath().normalize();
            Path rootPath = projectRootDir.toPath().toAbsolutePath().normalize();
            Path relativePath = rootPath.relativize(filePath);
            // Normalize path separators to "/"
            return relativePath.toString().replace(File.separator, "/");
        } catch (Exception e) {
            // If relative path calculation fails, return absolute path
            return file.getAbsolutePath();
        }
    }
    
    /**
     * Filter dependencies, keeping only project classes (matching project package prefix)
     */
    private static Set<String> filterProjectDependencies(Set<String> dependencies, String sourceClass) {
        if (projectPackagePrefix == null || projectPackagePrefix.isEmpty()) {
            // If no package prefix is set, return all dependencies
            return dependencies;
        }
        
        Set<String> filtered = new HashSet<>();
        for (String dependency : dependencies) {
            // Only keep dependencies starting with project package prefix
            if ((dependency.startsWith(projectPackagePrefix + ".") || dependency.equals(projectPackagePrefix)) && !dependency.equals(sourceClass)) {
                filtered.add(dependency);
            }
        }
        return filtered;
    }

    /**
     * Recursively find all src/main/java directories under the project root (supports multi-module projects)
     */
    private static List<File> findAllSourceDirs(File rootDir) {
        List<File> sourceDirs = new ArrayList<>();
        findSourceDirsRecursively(rootDir, sourceDirs);
        return sourceDirs;
    }
    
    /**
     * Helper method for recursively finding src/main/java directories
     */
    private static void findSourceDirsRecursively(File dir, List<File> sourceDirs) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // Check if current directory is src/main/java
        if (dir.getName().equals("java") && 
            dir.getParentFile() != null && dir.getParentFile().getName().equals("main") &&
            dir.getParentFile().getParentFile() != null && dir.getParentFile().getParentFile().getName().equals("src")) {
            sourceDirs.add(dir);
            return; // Found, do not go deeper to avoid duplicates
        }
        
        // Recursively search subdirectories
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    findSourceDirsRecursively(file, sourceDirs);
                }
            }
        }
    }

    /**
     * Recursively list all Java files
     */
    private static List<File> listJavaFilesRecursively(File dir) {
        List<File> javaFiles = new ArrayList<>();
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    javaFiles.addAll(listJavaFilesRecursively(file));
                } else if (file.getName().endsWith(".java")) {
                    javaFiles.add(file);
                }
            }
        }
        return javaFiles;
    }


    public static void loadStaticDependencies(CombinedTypeSolver combinedTypeSolver, File projectRootDir) {

        // Recursively find all microweaver-static-dependency directories
        List<File> dependencyDirs = findAllDependencyDirs(projectRootDir);
        
        if (dependencyDirs.isEmpty()) {
            System.out.println("No microweaver-static-dependency directory found, executing Maven command to download dependencies...");
            
            // Find all directories containing pom.xml
            System.out.println("Searching for all directories containing pom.xml...");
            List<File> pomDirs = findAllPomDirs(projectRootDir);
            System.out.println("Found " + pomDirs.size() + " directories containing pom.xml");
            
            // Execute Maven command in each directory containing pom.xml
            for (File pomDir : pomDirs) {
                System.out.println("Executing Maven command in directory: " + pomDir.getAbsolutePath());
                executeMavenCopyDependencies(pomDir);
            }
            
            // Re-search dependency directories
            dependencyDirs = findAllDependencyDirs(projectRootDir);
        }
        
        // Collect JAR files from all dependency directories and deduplicate (by file name)
        // Use Map for deduplication: key is file name, value is File object
        // This avoids loading duplicate JAR files (different modules may have the same version of dependency)
        // Note: if different modules use different versions of the same dependency, only the first encountered version is kept
        Map<String, File> uniqueJarFiles = new HashMap<>();
        int totalJarCount = 0;
        
        System.out.println("Found " + dependencyDirs.size() + " microweaver-static-dependency directories:");
        for (File dependencyDir : dependencyDirs) {
            File[] jars = dependencyDir.listFiles((dir, name) -> name.endsWith(".jar"));
            int jarCount = jars != null ? jars.length : 0;
            totalJarCount += jarCount;
            System.out.println("  - " + dependencyDir.getAbsolutePath() + " (" + jarCount + " JAR files)");
            if (jars != null) {
                for (File jar : jars) {
                    // Use file name as key for deduplication
                    // Same version dependencies will have the same file name (e.g. commons-lang3-3.12.0.jar)
                    String jarName = jar.getName();
                    if (!uniqueJarFiles.containsKey(jarName)) {
                        uniqueJarFiles.put(jarName, jar);
                    }
                }
            }
        }
        
        System.out.println("Total JAR files before dedup: " + totalJarCount);
        System.out.println("Unique JAR files after dedup: " + uniqueJarFiles.size());
        if (totalJarCount > uniqueJarFiles.size()) {
            System.out.println("Removed " + (totalJarCount - uniqueJarFiles.size()) + " duplicate JAR files");
        }
        
        // Actually load JAR files into CombinedTypeSolver, and collect successfully loaded dependency info
        int loadedCount = 0;
        int failedCount = 0;
        List<String> loadedDependencies = new ArrayList<>();
        
        for (File jarFile : uniqueJarFiles.values()) {
            try {
                JarTypeSolver jarTypeSolver = new JarTypeSolver(jarFile);
                jarTypeSolvers.add(jarTypeSolver);
                combinedTypeSolver.add(jarTypeSolver);
                loadedCount++;
                // Collect successfully loaded dependency names
                loadedDependencies.add(jarFile.getName());
            } catch (Exception e) {
                failedCount++;
                logError(null, "Failed to load JAR file", jarFile.getName(), e, false);
            }
        }
        
        System.out.println("Successfully loaded " + loadedCount + " JAR files into type resolver");
        if (failedCount > 0) {
            System.out.println("Failed to load " + failedCount + " JAR files");
        }
        
//         // Save successfully loaded dependency names to JSON file
//         try {
//             Gson gson = new GsonBuilder().setPrettyPrinting().create();
//             String json = gson.toJson(loadedDependencies);
//
//             try (FileWriter writer = new FileWriter("output/static-dependencies.json")) {
//                 writer.write(json);
//                 System.out.println("Static dependency info saved to static-dependencies.json (" + loadedDependencies.size() + " dependencies)");
//             }
//         } catch (IOException e) {
//             logError(null, "Failed to save static dependency info", null, e, false);
//         }
    }


    /**
     * Execute Maven dependency:copy-dependencies command
     * @param dir Directory to execute command in
     */
    private static void executeMavenCopyDependencies(File dir) {
        try {
            File microweaverStaticDir = new File(dir, "microweaver-static-dependency");
            if (microweaverStaticDir.exists() && microweaverStaticDir.isDirectory()) {
                // If microweaver-static-dependency directory already exists, skip Maven command execution
                return;
            }
            List<String> cmd = System.getProperty("os.name").toLowerCase().contains("win")
                ? List.of("cmd", "/c", "mvn", "dependency:copy-dependencies", 
                          "-DoutputDirectory=microweaver-static-dependency",
                          "-DincludeScope=compile", 
                          "-DexcludeTransitive=false",
                          "-q")  // -q parameter reduces output
                : List.of("mvn", "dependency:copy-dependencies", 
                          "-DoutputDirectory=microweaver-static-dependency",
                          "-DincludeScope=compile",
                          "-DexcludeTransitive=false",
                          "-q");
            Process process = new ProcessBuilder(cmd)
                .directory(dir)
                .inheritIO()  // Let Maven output display directly in console
                .start();
            int exitCode = process.waitFor();
            
            if (exitCode != 0) {
                System.out.println("Warning: Maven command in directory " + dir.getAbsolutePath() + " returned non-zero exit code: " + exitCode);
            }
        } catch (Exception e) {
            logError(null, "Maven command execution failed", dir.getAbsolutePath(), e, false);
        }
    }
    
    /**
     * Recursively find all directories containing pom.xml
     * @param rootDir Root directory
     * @return List of directories containing pom.xml
     */
    private static List<File> findAllPomDirs(File rootDir) {
        List<File> pomDirs = new ArrayList<>();
        findPomDirsRecursively(rootDir, pomDirs);
        return pomDirs;
    }
    
    /**
     * Helper method for recursively finding directories containing pom.xml
     * @param dir Current directory
     * @param pomDirs Result list
     */
    private static void findPomDirsRecursively(File dir, List<File> pomDirs) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // Check if current directory contains pom.xml
        File pomFile = new File(dir, "pom.xml");
        if (pomFile.exists() && pomFile.isFile()) {
            pomDirs.add(dir);
        }
        
        // Recursively search subdirectories
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    // Skip common build and dependency directories
                    String dirName = file.getName();
                    if (!dirName.equals("target") && 
                        !dirName.equals(".git") && 
                        !dirName.equals(".svn") &&
                        !dirName.equals("node_modules") &&
                        !dirName.equals(".idea") &&
                        !dirName.equals(".vscode") &&
                        !dirName.equals("microweaver-static-dependency")) {
                        findPomDirsRecursively(file, pomDirs);
                    }
                }
            }
        }
    }
    
    /**
     * Recursively find all microweaver-static-dependency directories under project root
     */
    private static List<File> findAllDependencyDirs(File rootDir) {
        List<File> dependencyDirs = new ArrayList<>();
        findDependencyDirsRecursively(rootDir, dependencyDirs);
        return dependencyDirs;
    }

    /**
     * Helper method for recursively finding microweaver-static-dependency directories
     */
    private static void findDependencyDirsRecursively(File dir, List<File> dependencyDirs) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // Check if current directory is microweaver-static-dependency
        if (dir.getName().equals("microweaver-static-dependency")) {
            dependencyDirs.add(dir);
            return; // Found, do not go deeper to avoid duplicates
        }
        
        // Recursively search subdirectories
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    findDependencyDirsRecursively(file, dependencyDirs);
                }
            }
        }
    }

    
    /**
     * Unified error logging function
     * @param javaFile The Java file with error (can be null)
     * @param errorType Error type description (e.g. "File analysis failed", "Type analysis failed", etc.)
     * @param details Error details (e.g. type name, expression, etc.)
     * @param e Exception object (can be null)
     * @param includeStackTrace Whether to include stack trace
     */
    private static void logError(File javaFile, String errorType, String details, Exception e, boolean includeStackTrace) {
        if (e != null && (e instanceof IllegalStateException || e instanceof UnsupportedOperationException)) {
            return;
        }
        StringBuilder errorMsg = new StringBuilder();
        
        // Add file information
        if (javaFile != null) {
            errorMsg.append("File: ").append(javaFile.getAbsolutePath());
            errorMsg.append(" | ");
        }
        
        // Add error type
        errorMsg.append(errorType);
        
        // Add details
        if (details != null && !details.isEmpty()) {
            errorMsg.append(": ").append(details);
        }
        
        // Add exception information
        if (e != null) {
            // Add exception type and message
            errorMsg.append(" | Exception type: ").append(e.getClass().getSimpleName());
            errorMsg.append(" | Error: ").append(e.getMessage());
        }
        
        String errorMessage = errorMsg.toString();
        
        // Output to console
        System.err.println(errorMessage);
        
        // Add to error collector
        errorMessages.offer(errorMessage);
        
        // Add stack trace if needed
        if (includeStackTrace && e != null) {
            StringBuilder stackTrace = new StringBuilder();
            stackTrace.append(errorMessage).append("\n");
            for (StackTraceElement element : e.getStackTrace()) {
                stackTrace.append("    at ").append(element.toString()).append("\n");
            }
            errorMessages.offer(stackTrace.toString());
        }
    }

    private static void deleteDependencyDirsRecursively(File dir) {
        if (dir == null || !dir.exists() || !dir.isDirectory()) {
            return;
        }
        
        // Check if current directory is microweaver-static-dependency
        if (dir.getName().equals("microweaver-static-dependency")) {
            // Java 7+ can use java.nio.file.Files.delete(Path) to recursively delete entire directory
            try {
                java.nio.file.Files.walk(dir.toPath())
                    .sorted(java.util.Comparator.reverseOrder())
                    .forEach(path -> {
                        try {
                            java.nio.file.Files.delete(path);
                        } catch (Exception e) {
                            // Ignore deletion errors
                        }
                    });
            } catch (Exception e) {
                // Ignore exceptions
            }
            return; // Found, do not go deeper to avoid duplicates
        }
        
        // Recursively search subdirectories
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    deleteDependencyDirsRecursively(file);
                }
            }
        }
    }

    /**
     * Unified output of class information to a JSON file
     */
    private static void writeClassInfoToFile(Map<String, ClassInfo> classInfoMap, String fileName) {
        // Save as JSON, disable HTML escaping to display < and > directly
        Gson gson = new GsonBuilder()
                .setPrettyPrinting()
                .disableHtmlEscaping()
                .create();
        String json = gson.toJson(classInfoMap);

        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write(json);
            System.out.println(fileName + " saved (" + classInfoMap.size() + " classes)");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
}