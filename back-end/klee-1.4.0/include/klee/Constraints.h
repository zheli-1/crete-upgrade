//===-- Constraints.h -------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_CONSTRAINTS_H
#define KLEE_CONSTRAINTS_H

#include "klee/Expr.h"

#if defined(CRETE_CONFIG)
#include "klee/util/Assignment.h"

#include <boost/unordered_set.hpp>
#endif

// FIXME: Currently we use ConstraintManager for two things: to pass
// sets of constraints around, and to optimize constraints. We should
// move the first usage into a separate data structure
// (ConstraintSet?) which ConstraintManager could embed if it likes.
namespace klee {

class ExprVisitor;
  
#if defined(CRETE_CONFIG)
// <symbolic-array, index-within-array>
typedef std::pair<const Array*, uint64_t> constraint_dependency_elem_ty;
typedef boost::unordered_set<constraint_dependency_elem_ty> constraint_dependency_ty;

class CreteConstraintDependency
{
friend class ConstraintManager;
public:
    CreteConstraintDependency() {}

    static void get_expr_cs_deps(ref<Expr> expr,
            constraint_dependency_ty &deps,
            uint64_t caller_num = 0);

protected:
    void add_dep(ref<Expr> e);
    const constraint_dependency_ty& get_last_cs_deps() const
    {
        return m_last_cs_deps;
    }
    const constraint_dependency_ty& get_complete_cs_deps() const
    {
        return m_complete_cs_deps;
    }

    void print_deps() const;

private:
    constraint_dependency_ty m_last_cs_deps;
    constraint_dependency_ty m_complete_cs_deps;
};
#endif //defined(CRETE_CONFIG)

class ConstraintManager {
public:
  typedef std::vector< ref<Expr> > constraints_ty;
  typedef constraints_ty::iterator iterator;
  typedef constraints_ty::const_iterator const_iterator;

  ConstraintManager() {}

  // create from constraints with no optimization
  explicit
  ConstraintManager(const std::vector< ref<Expr> > &_constraints) :
    constraints(_constraints) {}

  ConstraintManager(const ConstraintManager &cs) : constraints(cs.constraints)
#if defined(CRETE_CONFIG)
  ,m_crete_cs_deps(cs.m_crete_cs_deps)
#endif
  {}

  typedef std::vector< ref<Expr> >::const_iterator constraint_iterator;

  // given a constraint which is known to be valid, attempt to 
  // simplify the existing constraint set
  void simplifyForValidConstraint(ref<Expr> e);

  ref<Expr> simplifyExpr(ref<Expr> e) const;

  void addConstraint(ref<Expr> e);
  
  bool empty() const {
    return constraints.empty();
  }
  ref<Expr> back() const {
    return constraints.back();
  }
  constraint_iterator begin() const {
    return constraints.begin();
  }
  constraint_iterator end() const {
    return constraints.end();
  }
  size_t size() const {
    return constraints.size();
  }

  bool operator==(const ConstraintManager &other) const {
    return constraints == other.constraints;
  }
  
private:
  std::vector< ref<Expr> > constraints;

  // returns true iff the constraints were modified
  bool rewriteConstraints(ExprVisitor &visitor);

  void addConstraintInternal(ref<Expr> e);

#if defined(CRETE_CONFIG)
public:
  const constraint_dependency_ty& get_last_cs_deps() const
  {
      return m_crete_cs_deps.get_last_cs_deps();
  }

  const constraint_dependency_ty& get_complete_cs_deps() const
  {
      return m_crete_cs_deps.get_complete_cs_deps();
  }

  void print_constraints() const;

private:
  CreteConstraintDependency m_crete_cs_deps;
#endif
};

}

#endif /* KLEE_CONSTRAINTS_H */
